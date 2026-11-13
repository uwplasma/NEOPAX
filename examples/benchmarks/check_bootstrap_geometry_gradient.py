"""Isolate terminal-bootstrap geometry AD/FD at one common plasma state.

One baseline transport prefix is followed by geometry-only endpoint builds.
The AD path uses production corrected-Upar pullbacks, the recorded NTX table
transpose, and the VMEX-state payload transpose. No reverse transport is run.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import benchmark_transport_realtime_geometry_forward_fd as fd
from NEOPAX._geometry_autodiff import geometry_payload_pullback_from_param_vector_raw_block_transpose
from NEOPAX._reverse_ad_initial_er import (
    fold_recorded_ntx_scan_database_bar_groups_into_support,
    realtime_geometry_reverse_support_payload_for_runtime,
    split_recorded_ntx_database_runtime,
)
from NEOPAX._reverse_ad_transport import (
    _database_bootstrap_interpolation_bar,
    _float_delta_tree_like,
    _sanitize_float_delta_bar_tree,
    bootstrap_current_softmax_abs_scaled,
    bootstrap_current_softmax_abs_value_and_upar_bar,
)

DEFAULT_CONFIG = ROOT / 'examples/benchmarks/Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_black_box.toml'


def _tree_dot(bar, tangent):
    pairs = jax.tree_util.tree_leaves(jax.tree_util.tree_map(
        lambda b, t: jnp.vdot(b, t).real, bar, tangent,
    ))
    result = float(jax.device_get(sum(pairs)))
    if not math.isfinite(result):
        raise FloatingPointError('Nonfinite bootstrap directional derivative.')
    return result


def _steps(text):
    values = tuple(float(value) for value in text.split(','))
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise argparse.ArgumentTypeError('Step scales must be finite positive numbers.')
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError('Step scales must be distinct.')
    return values


def _fixed_state_fd(final_state, build_runtime, steps, value_fn):
    """Hold state identity fixed and evaluate one pair of runtimes per step."""
    for h in steps:
        minus = float(jax.device_get(value_fn(final_state, build_runtime(-h))))
        plus = float(jax.device_get(value_fn(final_state, build_runtime(h))))
        if not math.isfinite(minus) or not math.isfinite(plus):
            raise FloatingPointError(f'Nonfinite fixed-state bootstrap endpoint at h={h:.16e}.')
        gradient = (plus - minus) / (2 * h)
        if not math.isfinite(gradient):
            raise FloatingPointError(f'Nonfinite fixed-state bootstrap FD at h={h:.16e}.')
        yield {'fd_step': h, 'objective_minus': minus, 'objective_plus': plus,
               'gradient_fd': gradient}


def _bootstrap_geometry_ad(runtime, final_state, bundle, config, parameter):
    segment, owner = split_recorded_ntx_database_runtime(runtime)
    local_runtime = segment.runtime
    payload = realtime_geometry_reverse_support_payload_for_runtime(local_runtime)
    model = getattr(local_runtime.models.flux, 'neoclassical_model', local_runtime.models.flux)
    upar = model.evaluate_momentum_corrected_upar_only(final_state)
    value, upar_bar = bootstrap_current_softmax_abs_value_and_upar_bar(
        final_state, local_runtime, {'Upar': upar},
    )
    _, geometry_bar = model.pullback_momentum_corrected_upar_state_geometry_by_radius(
        final_state, upar_bar, payload['geometry'],
    )
    database_bar = _database_bootstrap_interpolation_bar(
        payload['database'], final_state, upar_bar, mode='established',
        table_pullback=model.pullback_momentum_corrected_upar_database_by_radius,
        coordinate_pullback=model.pullback_momentum_corrected_upar_database_coordinates_by_radius,
        sparse_pullback=model.pullback_momentum_corrected_upar_database_support_legacy_sparse_by_radius,
    )
    direct = {
        'geometry': _sanitize_float_delta_bar_tree(payload['geometry'], geometry_bar),
        'database': _float_delta_tree_like(payload['database']),
    }
    table = {
        'geometry': _float_delta_tree_like(payload['geometry']),
        'database': _sanitize_float_delta_bar_tree(payload['database'], database_bar),
    }
    print('[bootstrap-geometry] production recorded NTX transpose; two component rows', flush=True)
    (folded,) = fold_recorded_ntx_scan_database_bar_groups_into_support(owner, ((direct, table),))
    nc = config['neoclassical']
    state_bars = geometry_payload_pullback_from_param_vector_raw_block_transpose(
        bundle['context'], jnp.asarray([config.get('geometry', {}).get('vmec_param_delta', 0.0)]),
        fd._geometry_param_specs_from_parameter_name(parameter), folded,
        combined_payload=True, payload_kind='ntx_scan_runtime',
        scan_rho=nc['ntx_scan_rho'], scan_surface_backend=nc.get('ntx_scan_surface_backend', 'vmec'),
        n_r=bundle['n_r'], n_theta=int(nc['ntx_scan_n_theta']),
        n_zeta=int(nc['ntx_scan_n_zeta']), n_xi=int(nc['ntx_scan_n_xi']),
        surface_backend=nc.get('ntx_exact_surface_backend', 'booz'),
        raw_block_solve=bundle['raw_block_solve'], return_state_bars=True,
    )
    gradients = tuple(_tree_dot(jax.tree_util.tree_map(lambda b: b[i], state_bars),
                                bundle['state_tangent']) for i in range(2))
    return {
        'upar_only_objective': float(jax.device_get(value)),
        'direct_geometry_ad': gradients[0], 'ntx_database_ad': gradients[1],
        'fixed_state_geometry_ad': sum(gradients),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--parameter', default='RBC:1:0')
    parser.add_argument('--accepted-step-limit', type=int, default=16)
    parser.add_argument('--step-scales', type=_steps, default=_steps('3,1,0.3333333333333333'))
    parser.add_argument('--fd-rel-step', type=float, default=3e-7)
    parser.add_argument('--fd-abs-step', type=float, default=1e-10)
    parser.add_argument('--out-dir', type=Path, default=ROOT / 'outputs/bootstrap_geometry_gradient')
    args = parser.parse_args(argv)
    if args.accepted_step_limit <= 0:
        parser.error('--accepted-step-limit must be positive.')
    if (not math.isfinite(args.fd_rel_step) or args.fd_rel_step <= 0 or
            not math.isfinite(args.fd_abs_step) or args.fd_abs_step <= 0):
        parser.error('FD step settings must be finite and positive.')
    if args.parameter in fd.PARAMETER_ORDER:
        parser.error('Select a geometry parameter, e.g. RBC:1:0 or ZBS:1:0.')
    if jax.default_backend() != 'gpu':
        parser.error('This diagnostic requires a GPU; no CPU physics fallback is enabled.')
    if not jax.config.x64_enabled:
        parser.error('Set JAX_ENABLE_X64=true for this derivative comparison.')
    config = fd._prepare_benchmark_config(
        args.config.resolve(), device=None, ntx_exact_derivative_mode='direct',
        radau_jacobian_reuse_mode='legacy',
    )
    if str(config.get('geometry', {}).get('vmec_lane', '')).lower() != 'ad':
        parser.error('Use the benchmark TOML with geometry.vmec_lane="ad".')
    if config.get('neoclassical', {}).get('flux_model') != 'ntx_scan_runtime':
        parser.error('This diagnostic requires the ntx_scan_runtime model.')
    nc = config['neoclassical']
    nc.update(ntx_scan_record_primal=True, ntx_scan_coefficient_reverse_mode='structured')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    baseline_delta = float(config['geometry'].get('vmec_param_delta', 0.0))
    bundle = fd._frozen_linearized_vmec_geometry_bundle(
        config, args.parameter, baseline_delta=baseline_delta, retain_raw_solve=True,
    )
    runtime, initial = fd._runtime_for_frozen_linearized_geometry_step(config, bundle, step_scale=0.0)
    profiles = fd._baseline_profile_cfg(config)
    values = jnp.asarray([fd._profile_cfg_scalar_value(profiles, name) for name in fd.PARAMETER_ORDER])
    initial = fd._profile_state_from_values(
        values, config=config, initial_er_root_ad='jax_selected_root',
        runtime=runtime, baseline_state=initial, profile_cfg=profiles,
    )
    print('[bootstrap-geometry] ONE baseline transport prefix; all endpoints hold its final state fixed', flush=True)
    _, _, _, rollout, final = fd._schedule_rollout(
        config, runtime, initial, accepted_step_limit=args.accepted_step_limit,
        capture_newton_setup=True,
    )
    trace = fd._adaptive_rollout_diagnostics(rollout)
    if trace['accepted_count'] != args.accepted_step_limit or trace['failed']:
        raise RuntimeError('The baseline did not reach the requested accepted-step prefix.')
    baseline = float(jax.device_get(bootstrap_current_softmax_abs_scaled(final, runtime)))
    print(f'[bootstrap-geometry] baseline_bootstrap={baseline:.16e}', flush=True)
    np.savez_compressed(args.out_dir / 'baseline_final_state.npz',
                        density=np.asarray(final.density), pressure=np.asarray(final.pressure),
                        Er=np.asarray(final.Er))
    report = dict(parameter=args.parameter, accepted_step_limit=args.accepted_step_limit,
                  jax_version=jax.__version__, config=str(args.config.resolve()),
                  baseline_objective=baseline, baseline_trace=trace, rows=[])
    summary_path = args.out_dir / 'summary.json'
    summary_path.write_text(json.dumps(report, indent=2))
    ad = _bootstrap_geometry_ad(runtime, final, bundle, config, args.parameter)
    ad['upar_only_primal_absolute_difference'] = abs(ad['upar_only_objective'] - baseline)
    ad['upar_only_primal_relative_difference'] = (
        ad['upar_only_primal_absolute_difference'] / abs(baseline) if baseline != 0 else None
    )
    report.update(ad)
    summary_path.write_text(json.dumps(report, indent=2))
    print('[bootstrap-geometry] AD ' + json.dumps(ad), flush=True)
    base_h = fd._fd_step(bundle['coefficient_value'], rel_step=args.fd_rel_step, abs_step=args.fd_abs_step)
    def build_runtime(delta):
        print(f'[bootstrap-geometry] geometry-only endpoint delta={delta:+.16e}; no transport solve', flush=True)
        return fd._runtime_for_frozen_linearized_geometry_step(config, bundle, step_scale=delta)[0]
    for row in _fixed_state_fd(final, build_runtime,
                               (base_h * s for s in args.step_scales), bootstrap_current_softmax_abs_scaled):
        row['gradient_ad'] = ad['fixed_state_geometry_ad']
        row['absolute_error'] = abs(row['gradient_ad'] - row['gradient_fd'])
        row['relative_error'] = (row['absolute_error'] / abs(row['gradient_fd'])
                                 if row['gradient_fd'] != 0 else None)
        report['rows'].append(row)
        summary_path.write_text(json.dumps(report, indent=2))
        error_text = f"{row['relative_error']:.6e}" if row['relative_error'] is not None else 'undefined (FD=0)'
        print(f"[bootstrap-geometry] h={row['fd_step']:.16e} "
              f"AD={row['gradient_ad']:.16e} FD={row['gradient_fd']:.16e} "
              f"absolute_error={row['absolute_error']:.6e} relative_error={error_text}", flush=True)
    with (args.out_dir / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report['rows'][0]))
        writer.writeheader()
        writer.writerows({key: f'{value:.16e}' if value is not None else '' for key, value in row.items()}
                         for row in report['rows'])
    print(f'[bootstrap-geometry] wrote {summary_path}', flush=True)


if __name__ == '__main__':
    main()
