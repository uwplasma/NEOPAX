"""Tiny diagnostic orchestration tests; no equilibrium, scan or transport solve."""
import argparse
import ast
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def helpers():
    path = Path(__file__).resolve().parents[1] / 'examples/benchmarks/check_bootstrap_geometry_gradient.py'
    module = ast.parse(path.read_text())
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    prefix = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    namespace = dict(jax=jax, jnp=jnp, np=np, math=math, argparse=argparse,
                     json=json, csv=csv, Path=Path, DEFAULT_CONFIG=Path('benchmark.toml'), ROOT=path.parents[2])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[prefix, *functions], type_ignores=[])),
                 str(path), 'exec'), namespace)
    return namespace


def test_same_final_state_and_single_baseline_across_all_fd_steps(tmp_path, monkeypatch):
    ns = helpers()
    monkeypatch.setattr(jax, 'default_backend', lambda: 'gpu')
    initial = object()
    final = SimpleNamespace(density=np.ones((2, 2)), pressure=np.ones((2, 2)), Er=np.ones(2))
    bundle = {'coefficient_value': 2.0}
    config = dict(geometry={'vmec_lane': 'ad'}, neoclassical={'flux_model': 'ntx_scan_runtime'})
    build = Mock(side_effect=lambda cfg, data, *, step_scale: (SimpleNamespace(delta=step_scale), initial))
    schedule = Mock(return_value=(None, None, None, object(), final))
    fd = SimpleNamespace(
        PARAMETER_ORDER=('n0', 'T0'),
        _prepare_benchmark_config=Mock(return_value=config),
        _frozen_linearized_vmec_geometry_bundle=Mock(return_value=bundle),
        _runtime_for_frozen_linearized_geometry_step=build,
        _baseline_profile_cfg=lambda cfg: {}, _profile_cfg_scalar_value=lambda cfg, name: 1.0,
        _profile_state_from_values=Mock(return_value=initial), _schedule_rollout=schedule,
        _adaptive_rollout_diagnostics=lambda rollout: {'accepted_count': 16, 'failed': False},
        _fd_step=lambda *a, **kw: 0.002,
    )
    seen = []
    def value(state, runtime):
        assert state is final
        seen.append(runtime.delta)
        return 2 + 7 * runtime.delta + runtime.delta ** 2
    ns.update(fd=fd, bootstrap_current_softmax_abs_scaled=value,
              _bootstrap_geometry_ad=Mock(return_value=dict(upar_only_objective=2.0,
                  direct_geometry_ad=3.0, ntx_database_ad=4.0, fixed_state_geometry_ad=7.0)))
    ns['main'](['--out-dir', str(tmp_path)])
    schedule.assert_called_once()
    assert schedule.call_args.args[2] is initial
    assert schedule.call_args.kwargs['accepted_step_limit'] == 16
    assert fd._frozen_linearized_vmec_geometry_bundle.call_args.kwargs['retain_raw_solve'] is True
    assert config['neoclassical']['ntx_scan_record_primal'] is True
    assert config['neoclassical']['ntx_scan_coefficient_reverse_mode'] == 'structured'
    np.testing.assert_allclose(seen, [0, -.006, .006, -.002, .002, -.002/3, .002/3])
    report = json.loads((tmp_path / 'summary.json').read_text())
    assert len(report['rows']) == 3
    assert all(row['relative_error'] < 1e-12 for row in report['rows'])
    assert (tmp_path / 'baseline_final_state.npz').is_file()
    assert (tmp_path / 'comparison.csv').is_file()
    for invalid in ('0', '-1', 'nan', 'inf', '1,1'):
        with pytest.raises(argparse.ArgumentTypeError):
            ns['_steps'](invalid)
    with pytest.raises(FloatingPointError, match='Nonfinite fixed-state'):
        list(ns['_fixed_state_fd'](final, lambda step: None, [1e-5], lambda *a: np.nan))


def test_ad_uses_production_table_fold_and_same_vmec_tangent():
    ns = helpers()
    geometry, database = {'g': jnp.asarray([1.])}, {'d': jnp.asarray([1.])}
    geometry_bar, database_bar = {'g': jnp.asarray([2.])}, {'d': jnp.asarray([5.])}
    final, upar, upar_bar, raw_solve, owner = (object() for _ in range(5))
    model = SimpleNamespace(
        evaluate_momentum_corrected_upar_only=Mock(return_value=upar),
        pullback_momentum_corrected_upar_state_geometry_by_radius=Mock(return_value=(None, geometry_bar)),
        pullback_momentum_corrected_upar_database_by_radius=Mock(),
        pullback_momentum_corrected_upar_database_coordinates_by_radius=Mock(),
        pullback_momentum_corrected_upar_database_support_legacy_sparse_by_radius=Mock(),
    )
    runtime = SimpleNamespace(models=SimpleNamespace(flux=SimpleNamespace(neoclassical_model=model)))
    bundle = dict(context=object(), raw_block_solve=raw_solve, n_r=51,
                  state_tangent={'R': jnp.asarray([.1, .2])})
    fold = Mock(side_effect=lambda scan_owner, groups: groups)
    payload_pb = Mock(return_value={'R': jnp.asarray([[2., 3.], [4., 5.]])})
    interpolation_pb = Mock(return_value=database_bar)
    ns.update(
        fd=SimpleNamespace(_geometry_param_specs_from_parameter_name=lambda p: (('RBC', 1, 0),)),
        split_recorded_ntx_database_runtime=lambda rt: (SimpleNamespace(runtime=runtime), owner),
        realtime_geometry_reverse_support_payload_for_runtime=lambda rt: dict(geometry=geometry, database=database),
        bootstrap_current_softmax_abs_value_and_upar_bar=Mock(return_value=(jnp.asarray(2.), upar_bar)),
        _database_bootstrap_interpolation_bar=interpolation_pb,
        _float_delta_tree_like=lambda tree: jax.tree_util.tree_map(jnp.zeros_like, tree),
        _sanitize_float_delta_bar_tree=lambda primal, bar: bar,
        fold_recorded_ntx_scan_database_bar_groups_into_support=fold,
        geometry_payload_pullback_from_param_vector_raw_block_transpose=payload_pb,
    )
    config = dict(neoclassical=dict(ntx_scan_rho=[.25, .5], ntx_scan_n_theta=25,
                                    ntx_scan_n_zeta=31, ntx_scan_n_xi=64))
    result = ns['_bootstrap_geometry_ad'](runtime, final, bundle, config, 'RBC:1:0')
    np.testing.assert_allclose([result['direct_geometry_ad'], result['ntx_database_ad'],
                                result['fixed_state_geometry_ad']], [.8, 1.4, 2.2])
    assert fold.call_count == 1 and fold.call_args.args[0] is owner
    direct, table = fold.call_args.args[1][0]
    np.testing.assert_array_equal(direct['database']['d'], [0.])
    np.testing.assert_array_equal(table['geometry']['g'], [0.])
    assert direct['geometry'] is geometry_bar and table['database'] is database_bar
    assert interpolation_pb.call_args.kwargs['mode'] == 'established'
    options = payload_pb.call_args.kwargs
    assert options['raw_block_solve'] is raw_solve and options['return_state_bars'] is True
    assert (options['n_theta'], options['n_zeta'], options['n_xi']) == (25, 31, 64)
    assert options['payload_kind'] == 'ntx_scan_runtime' and options['combined_payload'] is True
    with pytest.raises(FloatingPointError, match='Nonfinite bootstrap'):
        ns['_tree_dot']({'R': jnp.asarray([np.nan])}, {'R': jnp.asarray([1.])})
