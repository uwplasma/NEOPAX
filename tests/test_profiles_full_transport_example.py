"""Configuration and small plotting checks; no VMEC, NTX or transport solve."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tomllib

import h5py
import numpy as np
import pytest

from examples.optimization import (
    optimize_profiles_max_er_transition_bootstrap_net_power_initial_root_database_full_transport
    as example,
)


def config():
    value = example.load_config(example.TRANSPORT_CONFIG)
    value['geometry']['n_radial'] = 7
    return value


def parameters():
    return ({'n0':4.21,'T0':17.8,'density_shape_power':10.,'temperature_shape_power':2.},
            {'n0':2.,'T0':25.,'density_shape_power':12.,'temperature_shape_power':10.835566224074642})


def test_defaults_select_new_root_terms_and_51_surface_whe_equilibrium():
    args = example.parser().parse_args([])
    benchmark = example.ROOT/'examples/benchmarks/Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_black_box.toml'
    source = tomllib.loads(benchmark.read_text())
    reference = (example.ROOT/source['geometry']['vmec_input_file']).resolve()
    assert args.seed_input.name == 'input.QI_nfp2_newNT_opt_hires'
    assert args.seed_input.is_file()
    # Only radial solver staging differs from the benchmark's *_true input.
    staging = ('NS_ARRAY', 'NITER_ARRAY', 'FTOL_ARRAY')
    physical_lines = lambda path:[line.strip() for line in path.read_text().splitlines()
                                 if not line.strip().startswith(staging)]
    assert physical_lines(args.seed_input) == physical_lines(reference)
    ns_line = next(line for line in args.seed_input.read_text().splitlines() if line.strip().startswith('NS_ARRAY'))
    assert [int(token) for token in ns_line.split('=')[1].split()] == [24,51]
    assert args.transition_location_objective and not args.root_objectives
    assert args.bootstrap_penalty and args.net_power and not args.max_er
    objects = [term[0] for term in example.active_terms(args)]
    assert example.opt.transport.Er_transition_left not in objects
    assert example.opt.transport.Er_transition_right not in objects
    assert example.opt.transport.Er_transition_location_moment in objects
    assert example.opt.transport.net_total_power_volume_average_mw_m3 in objects
    assert any(getattr(obj,'label',None)=='Er_transition_strength_deficit' for obj in objects)
    prepared = example.transport_config(args)
    assert prepared['geometry']['vmec_input_file'] == str(args.seed_input.resolve())
    assert prepared['geometry']['n_radial'] == 51
    assert prepared['energy_grid']['n_x'] == 4
    assert prepared['transport_solver']['t_final'] == 2.


def test_saved_parameter_plotting_does_not_initialize_physics(tmp_path, monkeypatch):
    before, after = parameters()
    value = config()
    original = copy.deepcopy(value)
    summary = dict(profile_plot_config=value, initial_physical_profiles=before,
                   optimized_physical_profiles=after)
    (tmp_path/'optimization_summary.json').write_text(json.dumps(summary))

    def forbidden(*args, **kwargs):
        pytest.fail('Profile-only plotting must not initialize optimization or transport')

    monkeypatch.setattr(example.opt,'full_transport_profile_least_squares_problem',forbidden)
    monkeypatch.setattr(example,'run_config',forbidden)
    assert example.main(['--out-dir',str(tmp_path),'--plot-profiles-only']) == 0
    assert value == original
    for suffix in ('png','pdf'):
        assert (tmp_path/f'initial_profiles_initial_vs_optimized.{suffix}').stat().st_size > 1000
    table = np.genfromtxt(tmp_path/'initial_optimized_initial_profiles.csv',delimiter=',',names=True)
    assert len(table) == 7
    assert len(table.dtype.names) == 17  # rho + n/T * four species * two cases.
    assert np.isfinite(table['initial_He_temperature']).all()
    assert (table['initial_He_temperature'] > 0).all()
    assert np.any(abs(table['optimized_D_density']-table['initial_D_density']) > 1e-4)
    assert np.any(abs(table['optimized_e_temperature']-table['initial_e_temperature']) > 1e-4)
    np.testing.assert_array_equal(table['initial_He_density'],np.zeros(7))


def test_final_comparison_reads_existing_reports_only(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Comparing saved reports must not run transport')

    monkeypatch.setattr(example,'run_config',forbidden)
    for i,label in enumerate(('initial','optimized')):
        folder = tmp_path/label/'transport'
        folder.mkdir(parents=True)
        with h5py.File(folder/'transport_solution.h5','w') as stream:
            stream['rho'] = np.linspace(.05,.95,7)
            stream['ts'] = np.array([0.,2.])
            stream['density'] = np.ones((2,4,7))*(i+1)
            stream['temperature'] = np.ones((2,4,7))*(i+2)
            stream['Er'] = np.broadcast_to(np.linspace(10.,-10.,7),(2,7))*(i+1)
    paths = example.write_final_profile_comparison(tmp_path)
    assert set(paths) == {'png','pdf'}
    assert all(Path(path).stat().st_size > 1000 for path in paths.values())


def test_main_saves_profile_comparison_before_running_transport_reports(tmp_path, monkeypatch):
    initial, optimized = parameters()
    labels = tuple(initial)
    scale = np.array(list(initial.values()))
    x0 = np.ones(4)
    xlast = np.array(list(optimized.values()))/scale
    calls = []
    geometry = object()
    problem = SimpleNamespace(x0=x0,x_scale=scale,parameter_count=4,parameter_labels=labels,
                              runtime=SimpleNamespace(geometry=geometry))
    evaluation = object()
    result = SimpleNamespace(x=xlast,cost=1.,optimality=.1,status=2,message='mock ftol')
    monkeypatch.setattr(example.opt,'full_transport_profile_least_squares_problem',lambda *a,**k:problem)
    monkeypatch.setattr(example,'report',lambda *a,**k:evaluation)
    monkeypatch.setattr(example.opt,'least_squares',lambda *a,**k:result)

    def compare(cfg,before,after,out,**kwargs):
        calls.append('input_plot')
        assert kwargs['geometry'] is geometry
        np.testing.assert_allclose(list(before.values()),list(initial.values()))
        np.testing.assert_allclose(list(after.values()),list(optimized.values()))
        return {'png':'input.png','pdf':'input.pdf','csv':'input.csv'}

    def report_transport(problem,x,out,label):
        calls.append(label)
        # The input-profile artifacts and summary survive a later report failure.
        summary = json.loads((out/'optimization_summary.json').read_text())
        assert summary['objectives']['transition_location_objective']
        assert not summary['objectives']['root_objectives']
        assert summary['initial_profile_comparison']['png'] == 'input.png'

    monkeypatch.setattr(example,'write_initial_profile_comparison',compare)
    monkeypatch.setattr(example,'write_transport_report',report_transport)
    monkeypatch.setattr(example,'write_final_profile_comparison',lambda out: calls.append('final_plot') or {'png':'final.png'})
    assert example.main(['--out-dir',str(tmp_path)]) == 0
    assert calls == ['input_plot','initial','optimized','final_plot']
    summary = json.loads((tmp_path/'optimization_summary.json').read_text())
    assert summary['final_profile_comparison'] == {'png':'final.png'}
