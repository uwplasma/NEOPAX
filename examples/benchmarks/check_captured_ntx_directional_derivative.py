"""Check the NTX derivative at important nodes of a captured AD16 run.

GPU only. Reuses saved VMEC state and numerical surface/channel directions.
No equilibrium, ambipolar root, transport integration or complete NTX scan.
Compares raw-solver JVP, production database VJP and local finite differences.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from pathlib import Path
import sys
import tomllib

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SURFACE_FIELDS = ('b_cos', 'jacobian_cos', 'b_sub_theta_cos', 'b_sub_zeta_cos',
                  'b_sup_theta_cos', 'b_sup_zeta_cos', 'b0')
COEFFICIENTS = (('D11', 0), ('D13', 2), ('D33', 3))


def saved(directory, pattern):
    paths = list(directory.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f'Expected one {pattern} in {directory}, found {len(paths)}')
    with np.load(paths[0], allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def fields(arrays, prefix):
    return {key[len(prefix):]: value for key, value in arrays.items()
            if key.startswith(prefix) and '.' not in key[len(prefix):]}


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--parameter', choices=('RBC:1:0', 'ZBS:1:0'), default='RBC:1:0')
    parser.add_argument('--points', type=point_indices, default=point_indices('4:4:2,4:4:3,4:4:5'))
    args = parser.parse_args()
    terminal = saved(args.run_dir/'ad_snapshots', 'ad_terminal_*.npz')
    capture = saved(args.run_dir/'ad_snapshots', 'ad_compact_geometry_*.npz')
    directions = captured_direction(saved(args.run_dir/'ad_snapshots', 'ad_compact_tangents_*.npz'), args.parameter)
    fd = json.loads((args.run_dir/f"fd_{args.parameter.replace(':', '_')}"/'summary.json').read_text())
    h0 = float(fd['fd_step'])
    if not np.isfinite(h0) or h0 <= 0:
        parser.error('Saved FD step must be finite and positive')
    with (args.run_dir/'config.toml').open('rb') as stream:
        config = tomllib.load(stream)
    nc = config['neoclassical']
    for radius, nu_index, field_index in args.points:
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
    from NEOPAX._transport_flux_models import _build_ntx_field_channels
    from benchmark_transport_reverse_ad_only import _geometry_context_from_config

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
        def adjoint(values, direction, nu):
            prepared = ntx.prepare_monoenergetic_system(local_surface(values), grid)
            implicit_primal = _prepared_implicit_vjp_primal(prepared, nu, values[7])[0]
            case_bars, geometry_bars = native_ad.pullback_prepared_database_coefficients_to_vmec_primitive_multi_rhs(
                prepared, ntx.MonoenergeticCase(nu_hat=nu, epsi_hat=values[7]), basis)
            _, pullback = jax.vjp(lambda x: geometry_on_grid(local_surface(x), grid), values)
            def contract(index):
                one = jax.tree_util.tree_map(lambda value: value[index], geometry_bars)
                (bars,) = pullback(one)
                return sum(jnp.vdot(bar, dot).real for bar, dot in zip(bars, direction, strict=True)) + case_bars.epsi_hat[index]*direction[7]
            return implicit_primal, jax.vmap(contract)(jnp.arange(3))

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
            implicit_primal, reverse = map(np.asarray, jax.block_until_ready(adjoint(values, direction, nu)))
            if not all(np.isfinite(value).all() for value in (baseline, jvp_primal, implicit_primal, tangent, reverse)):
                raise FloatingPointError('Nonfinite coefficient or derivative')
            for scale in (10., 3., 1.):
                h = h0*scale
                minus = np.asarray(jax.block_until_ready(primal_jit(tuple(x-h*d for x,d in zip(values,direction,strict=True)), nu)))
                plus = np.asarray(jax.block_until_ready(primal_jit(tuple(x+h*d for x,d in zip(values,direction,strict=True)), nu)))
                if not np.isfinite(minus).all() or not np.isfinite(plus).all():
                    raise FloatingPointError('Nonfinite local coefficient FD endpoint')
                for channel, (name, index) in enumerate(COEFFICIENTS):
                    fd_value = float((plus[index]-minus[index])/(2*h))
                    jvp_value, vjp_value = float(tangent[index]), float(reverse[channel])
                    relative = abs(vjp_value-jvp_value)/max(abs(jvp_value),np.finfo(float).tiny)
                    row = dict(radius_index=radius, nu_index=nu_index, field_index=field_index,
                        parameter=args.parameter, coefficient=name, fd_step=h, raw_solver_jvp=jvp_value,
                        production_vjp=vjp_value, gradient_fd=fd_value, jvp_vjp_relative_difference=relative,
                        forward_primal=float(baseline[index]), implicit_pullback_primal=float(implicit_primal[index]),
                        raw_jvp_primal=float(jvp_primal[index]),
                        raw_jvp_primal_relative_difference=float(abs(baseline[index]-jvp_primal[index])/max(abs(baseline[index]),np.finfo(float).tiny)),
                        forward_pullback_primal_relative_difference=float(abs(baseline[index]-implicit_primal[index])/max(abs(baseline[index]),np.finfo(float).tiny)),
                        fd_vjp_absolute_difference=abs(fd_value-vjp_value))
                    rows.append(row)
                    print('[captured-ntx] '+json.dumps(row), flush=True)
            with (args.out_dir/'comparison.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f'[captured-ntx] saved {len(rows)} rows in {args.out_dir}', flush=True)


if __name__ == '__main__':
    main()
