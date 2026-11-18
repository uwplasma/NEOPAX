"""Generic baseline diagnostic checks using NumPy only; no physical solves."""
from contextlib import redirect_stdout
from dataclasses import dataclass
import ast
import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("generic_baselines",
    ROOT/"examples/benchmarks/check_generic_ntx_database_baselines.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class GenericDatabaseBaselineTests(unittest.TestCase):
    def test_original_fd_setup_branch_is_unchanged_and_capture_exits_before_profiles(self):
        script = ROOT/'examples/benchmarks/benchmark_transport_realtime_geometry_forward_fd.py'
        tree = ast.parse(script.read_text())
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        start = next(i for i, node in enumerate(main.body) if isinstance(node, ast.Assign)
                     and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'profile_cfg')
        end = next(i for i, node in enumerate(main.body[start:], start) if isinstance(node, ast.Assign)
                   and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'profile_values')
        function = ast.parse('def setup():\n    pass\n').body[0]
        function.body = main.body[start:end] + ast.parse('return baseline_runtime, baseline_state').body
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            config_path = folder/'config.toml'
            config_path.write_text('[general]\nmode="transport"\n')
            args = SimpleNamespace(config=config_path, parameter='ZBS:1:0', geometry_fd_lane='frozen_linearized',
                initial_er_root_only_fd=False, geometry_input_only_dir=None, geometry_input_reference=None)
            state, geometry = np.ones(2), np.ones(1)
            geo = SimpleNamespace(build_ntx_runtime_scan_inputs_from_vmec_state=lambda *a, **kw: (geometry, ()))
            runtime = object()
            def build(config, bundle, *, step_scale):
                if args.geometry_input_only_dir is not None:
                    geo.build_ntx_runtime_scan_inputs_from_vmec_state('context', state, geometry)
                    self.fail('NTX construction must not be reached')
                return runtime, state
            save = Mock(return_value=folder/'capture.npz')
            namespace = dict(args=args, Path=Path, config={'geometry':{'vmec_param_delta':.125}},
                contextlib=contextlib, PARAMETER_ORDER=('n0',),
                _baseline_profile_cfg=lambda config: {},
                _frozen_linearized_vmec_geometry_bundle=Mock(return_value=object()),
                _runtime_for_frozen_linearized_geometry_step=Mock(side_effect=build),
                jnp=SimpleNamespace(asarray=np.asarray, float64=np.float64),
                save_gradient_audit=save, build_runtime_context=Mock())
            exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(script), 'exec'), namespace)
            result = namespace['setup']()
            self.assertIs(result[0], runtime)
            self.assertIs(result[1], state)
            save.assert_not_called()
            namespace['build_runtime_context'].assert_not_called()
            args.geometry_input_only_dir = folder/'capture'
            with patch.dict(sys.modules, NEOPAX=SimpleNamespace(_geometry_autodiff=geo),
                    check_generic_ntx_database_baselines=check), redirect_stdout(io.StringIO()):
                self.assertIsNone(namespace['setup']())
            save.assert_called_once()
            self.assertEqual((args.geometry_input_only_dir/'config.toml').read_text(), config_path.read_text())
            namespace['_frozen_linearized_vmec_geometry_bundle'].assert_called_with(
                namespace['config'], 'ZBS:1:0', baseline_delta=.125)
            self.assertEqual(namespace['_runtime_for_frozen_linearized_geometry_step'].call_count, 2)

    def test_terminal_capture_keeps_builders_unobserved_and_stops_only_after_scan(self):
        state, geometry, channels = np.arange(2.), np.ones(2), np.ones(1)
        build_geometry = Mock(return_value=geometry)
        build_scan = Mock(return_value=(channels, (state,)))
        geo = SimpleNamespace(_build_neopax_geometry_from_state=build_geometry,
            build_ntx_runtime_scan_inputs_from_vmec_state=build_scan)
        captured = {}
        with check.capture_scan_boundary(geo, captured):
            self.assertIs(geo._build_neopax_geometry_from_state, build_geometry)
            self.assertEqual(captured, {})
            result = geo._build_neopax_geometry_from_state('context', state)
            self.assertEqual(captured, {})
            geo.build_ntx_runtime_scan_inputs_from_vmec_state('context', state, result, rho_scan=[.5])
            self.fail('The NTX coefficient builder must not be reached')
        build_scan.assert_called_once_with('context', state, geometry, rho_scan=[.5])
        self.assertIs(captured['00_geometry_equilibrium'], state)
        self.assertIs(captured['01_geometry'], geometry)
        self.assertIs(captured['02_channels'], channels)
        self.assertIs(geo.build_ntx_runtime_scan_inputs_from_vmec_state, build_scan)
        with self.assertRaisesRegex(RuntimeError, 'pre-NTX scan boundary'):
            with check.capture_scan_boundary(geo, {}):
                pass
        with self.assertRaisesRegex(ValueError, 'original failure'):
            with check.capture_scan_boundary(geo, {}):
                raise ValueError('original failure')
        self.assertIs(geo.build_ntx_runtime_scan_inputs_from_vmec_state, build_scan)

    def test_common_boundary_comparison_aligns_detailed_and_terminal_stage_numbers(self):
        reference = {"trace['02_geometry_equilibrium']['R_cos']": np.ones(1),
            "trace['43_geometry']['a_b']": np.asarray(1.),
            "trace['80_channels']['b00']": np.ones(2),
            "trace['81_surfaces'][0]['b_cos']": np.ones(2),
            "trace['42_boozer_0_outputs']['bmnc_b']": np.ones(2)}
        captured = {key.replace('02_geometry_equilibrium', '00_geometry_equilibrium')
            .replace('43_geometry', '01_geometry').replace('80_channels', '02_channels')
            .replace('81_surfaces', '03_surfaces'): value for key, value in reference.items()}
        a, b = map(check.common_input_boundaries, (reference, captured))
        self.assertEqual(len(a), 4)
        with redirect_stdout(io.StringIO()):
            rows = check.compare(a, b, 'terminal')
        self.assertTrue(all(row['arrays_match'] and row['bits_match'] for row in rows))
        b["trace['01_geometry']['a_b']"] = np.nextafter(1., np.inf)
        with redirect_stdout(io.StringIO()):
            rows = check.compare(a, b, 'changed')
        self.assertEqual(check.first_changed_stage(rows), '01_geometry')
        with self.assertRaisesRegex(ValueError, 'no common'):
            check.common_input_boundaries({})

    def test_geometry_comparison_uses_the_original_fd_setup_and_preserves_interpolation(self):
        config={"general":{"mode":"transport"},"geometry":{"vmec_param_delta":0.125},
                "neoclassical":{"flux_model":"ntx_scan_runtime",
                                 "interpolation_mode":"preprocessed_3d_radial_ntss1d"}}
        canonical=Mock(return_value=object())
        bundle=object()
        forward=SimpleNamespace(_frozen_linearized_vmec_geometry_bundle=Mock(return_value=bundle),
                                _runtime_for_frozen_linearized_geometry_step=Mock(return_value=object()))
        actual=check.build_comparison_runtime(config,'ad','ZBS:1:0',canonical,forward)
        self.assertIs(actual,canonical.return_value)
        forward._frozen_linearized_vmec_geometry_bundle.assert_not_called()
        canonical.reset_mock()
        actual=check.build_comparison_runtime(config,'fd','ZBS:1:0',canonical,forward)
        canonical.assert_not_called()
        forward._frozen_linearized_vmec_geometry_bundle.assert_called_once_with(config,'ZBS:1:0',baseline_delta=0.125)
        forward._runtime_for_frozen_linearized_geometry_step.assert_called_once_with(config,bundle,step_scale=0.0)
        self.assertIs(actual,forward._runtime_for_frozen_linearized_geometry_step.return_value)
        traced=check.baseline_config(config,'fd',trace_inputs=True)
        self.assertEqual(traced['neoclassical']['interpolation_mode'],'preprocessed_3d_radial_ntss1d')
        self.assertEqual(config['general']['mode'],'transport')
        self.assertEqual(traced['general']['mode'],'ambipolarity')

    def test_pre_ntx_boundaries_align_when_only_one_builder_uses_the_scalar_solver(self):
        state=np.asarray([1.,2.])
        geometry=np.asarray([3.])
        channels=np.asarray([4.])
        geo=SimpleNamespace(_solve_state_for_single_param=Mock(return_value=state),
            _build_neopax_geometry_from_state=lambda *a,**kw: geometry,
            build_ntx_runtime_scan_inputs_from_vmec_state=lambda *a,**kw: (channels,()),
            _import_vmec_module=lambda *a: SimpleNamespace())
        api=SimpleNamespace(booz_xform_from_inputs=lambda **kw: {})
        def build(config,scalar_solve):
            value=geo._solve_state_for_single_param(None,0.) if scalar_solve else state
            field=geo._build_neopax_geometry_from_state(None,value)
            geo.build_ntx_runtime_scan_inputs_from_vmec_state(None,value,field)
        a=check.trace_inputs({},geo,api,lambda cfg:build(cfg,True),lambda c:c,record_equilibrium=False)
        b=check.trace_inputs({},geo,api,lambda cfg:build(cfg,False),lambda c:c,record_equilibrium=False)
        self.assertEqual(tuple(a),tuple(b))
        self.assertEqual(tuple(a)[0],'00_geometry_equilibrium')
        for name in ('00_geometry_equilibrium','01_geometry','02_channels','04_electric_fields'):
            np.testing.assert_array_equal(a[name],b[name])
        self.assertEqual(geo._solve_state_for_single_param.call_count,1)

    def test_parent_routes_ad_and_two_original_zbs_setups_to_input_trace(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/'trace'
            def complete(command,**kwargs):
                folder=Path(command[command.index('--out-dir')+1])
                self.assertEqual(command[command.index('--fd-geometry-parameter')+1],'ZBS:1:0')
                self.assertIn('--trace-inputs',command)
                np.savez(folder/'normalization_inputs_saved.npz',
                         **{"trace['00_geometry_equilibrium']['R_cos']":np.ones(1)})
                (folder/'metadata.json').write_text(json.dumps(dict(config_sha256='config',input_sha256='input',
                    reused_state_sha256=None,versions={},sources={})))
                return SimpleNamespace(returncode=0)
            with patch.object(sys,'argv',[str(check.__file__),'--out-dir',str(output),
                    '--trace-inputs','--fd-geometry-parameter','ZBS:1:0']), \
                 patch.object(check.subprocess,'run',side_effect=complete) as run,redirect_stdout(io.StringIO()) as log:
                check.main()
            self.assertEqual([call.args[0][call.args[0].index('--worker')+1] for call in run.call_args_list],['ad','fd','fd'])
            self.assertIn('first_changed_stage=none',log.getvalue())

    def test_vmec_primal_trace_separates_solve_output_from_zero_offset_assembly(self):
        params, cfg, state, mask = np.ones(2), np.ones(1), np.arange(2.), np.ones(2)
        implicit = SimpleNamespace(solve_implicit=Mock(return_value=state),
            solve_implicit_with_aux=Mock(return_value=(state, mask)))
        importer = lambda: implicit
        geo = SimpleNamespace(_import_vmec_jax_implicit=importer,
            _solve_state_for_single_param=lambda *a, **kw: None,
            _build_neopax_geometry_from_state=lambda *a, **kw: a[1],
            build_ntx_runtime_scan_inputs_from_vmec_state=lambda *a, **kw: (a[1], ()),
            _import_vmec_module=lambda *a: SimpleNamespace())
        api = SimpleNamespace(booz_xform_from_inputs=lambda **kw: {})
        def build(config, aux):
            module = geo._import_vmec_jax_implicit()
            result = module.solve_implicit_with_aux(params, cfg) if aux else module.solve_implicit(params, cfg)
            value = result[0] if aux else result
            if aux:
                self.assertIs(result[1], mask)
                value = value + 0.0 * np.ones_like(value)
            geometry = geo._build_neopax_geometry_from_state(None, value)
            geo.build_ntx_runtime_scan_inputs_from_vmec_state(None, value, geometry)
        traces = [check.trace_inputs({}, geo, api, lambda c: build(c, aux), lambda c: c,
            record_equilibrium=False, trace_vmec_primal=True) for aux in (False, True)]
        for trace in traces:
            self.assertIs(trace['00_vmec_solve_inputs']['params'], params)
            self.assertIs(trace['00_vmec_solve_inputs']['config'], cfg)
            self.assertIs(trace['01_vmec_solve_outputs'], state)
            np.testing.assert_array_equal(trace['02_geometry_equilibrium'], state)
        self.assertEqual(tuple(traces[0]), tuple(traces[1]))
        self.assertIs(geo._import_vmec_jax_implicit, importer)
        implicit.solve_implicit.assert_called_once_with(params, cfg)
        implicit.solve_implicit_with_aux.assert_called_once_with(params, cfg)

    def test_projection_trace_preserves_operands_and_restores_einsum(self):
        values, basis = np.ones((1,2,2)), np.ones((2,2,1))
        numerical = SimpleNamespace(einsum=np.einsum)
        magnetic = SimpleNamespace(bsubu=values)
        imported = SimpleNamespace(magnetic_fields=lambda **kw: magnetic)
        def geometry(*args, **kwargs):
            self.assertIs(geo._import_vmec_module('fields').magnetic_fields(), magnetic)
            return numerical.einsum('rtz,tzm->rm', values, basis)
        geo = SimpleNamespace(_solve_state_for_single_param=lambda *a, **kw: np.ones(1),
            _build_neopax_geometry_from_state=geometry, _import_vmec_module=lambda _: imported,
            build_ntx_runtime_scan_inputs_from_vmec_state=lambda *a, **kw: (a[2], ()))
        api = SimpleNamespace(booz_xform_from_inputs=lambda **kw: {})
        def runtime(_):
            state = geo._solve_state_for_single_param(None, 0.)
            result = geo._build_neopax_geometry_from_state(None, state)
            geo.build_ntx_runtime_scan_inputs_from_vmec_state(None, state, result)
        captured = check.trace_inputs({}, geo, api, runtime, lambda c: c, projection_module=numerical)
        self.assertIs(captured['02_vmex_magnetic_fields']['bsubu'], values)
        self.assertIs(captured['03_projection_00_inputs']['values'], values)
        self.assertIs(captured['03_projection_00_inputs']['basis'], basis)
        np.testing.assert_array_equal(captured['04_projection_00_outputs'], [[4.]])
        self.assertIs(numerical.einsum, np.einsum)

    def test_input_trace_observes_real_boundaries_and_restores_hooks(self):
        @dataclass
        class State:
            R_cos: np.ndarray
        @dataclass
        class Channels:
            a_b: np.ndarray
        state = State(np.array([1.]))
        frozen = State(np.array([2.]))
        calls = []
        norms = SimpleNamespace(volume=np.array(3.), vp=np.array([4.]))
        module = SimpleNamespace(vmec_force_norms_from_bcovar_dynamic=lambda **kw: norms)
        api = SimpleNamespace(booz_xform_from_inputs=lambda **kw: {'rmnc_b': kw['inputs'].R_cos})
        def solve(*args, **kwargs):
            calls.append('solve')
            return state
        def geometry(context, value, **kwargs):
            self.assertIs(geo._import_vmec_module('residue').vmec_force_norms_from_bcovar_dynamic(), norms)
            result = api.booz_xform_from_inputs(inputs=value, constants=SimpleNamespace(nfp=2), jit=True)
            return SimpleNamespace(R0=result['rmnc_b'])
        def scan(context, value, g, **kwargs):
            api.booz_xform_from_inputs(inputs=value, constants=SimpleNamespace(nfp=2), jit=True)
            return Channels(g.R0), (value,)
        geo = SimpleNamespace(_solve_state_for_single_param=solve,
            _build_neopax_geometry_from_state=geometry, _import_vmec_module=lambda _: module,
            build_ntx_runtime_scan_inputs_from_vmec_state=scan)
        def runtime(config):
            value = geo._solve_state_for_single_param('context', 0., lane='ad')
            g = geo._build_neopax_geometry_from_state('context', value, n_r=51)
            geo.build_ntx_runtime_scan_inputs_from_vmec_state('context', value, g, rho_scan=[.875])
            self.fail('NTX coefficient construction must not be reached')
        for seed in (None, frozen):
            captured = check.trace_inputs({}, geo, api, runtime, lambda c: {'Es': c.a_b*3}, seed)
            self.assertEqual(list(captured), ['00_equilibrium', '01_geometry_equilibrium', '02_volume_inputs',
                '03_boozer_0_inputs', '04_boozer_0_outputs', '05_geometry', '06_boozer_1_inputs',
                '07_boozer_1_outputs', '08_channels', '09_surfaces', '10_electric_fields'])
            expected = state if seed is None else seed
            self.assertIs(captured['00_equilibrium']['R_cos'], expected.R_cos)
            self.assertIs(captured['02_volume_inputs']['volume'], norms.volume)
            self.assertTrue(captured['03_boozer_0_inputs']['jit'])
            np.testing.assert_array_equal(captured['10_electric_fields']['Es'], expected.R_cos*3)
            self.assertIs(geo._solve_state_for_single_param, solve)
            self.assertIs(geo._build_neopax_geometry_from_state, geometry)
            self.assertIs(geo.build_ntx_runtime_scan_inputs_from_vmec_state, scan)
        self.assertEqual(calls, ['solve'])
        with self.assertRaisesRegex(RuntimeError, 'pre-NTX boundary'):
            check.trace_inputs({}, geo, api, lambda _: None, lambda _: {})
        self.assertIs(geo._solve_state_for_single_param, solve)

    def test_trace_reports_first_changed_boundary_and_exact_float_bits(self):
        reference = {"trace['00_equilibrium']['R_cos']": np.array([1.]),
            "trace['04_boozer_0_outputs']['rmnc_b']": np.array([2.]),
            "trace['10_electric_fields']['Es']": np.array([3.])}
        changed = {k: v.copy() for k,v in reference.items()}
        changed["trace['04_boozer_0_outputs']['rmnc_b']"][0] = np.nextafter(2., np.inf)
        with redirect_stdout(io.StringIO()):
            rows = check.compare(reference, changed, 'repeat')
        self.assertEqual(check.first_changed_stage(rows), '04_boozer_0_outputs')
        row = next(r for r in rows if not r['arrays_match'])
        self.assertEqual(row['reference_hex'], float(2.).hex())
        self.assertEqual(row['actual_hex'], float(np.nextafter(2., np.inf)).hex())
        self.assertEqual(check.first_changed_stage(check.compare(reference, reference, 'same')), 'none')
        changed["trace['11_extra']['value']"] = np.ones(1)
        with redirect_stdout(io.StringIO()):
            rows = check.compare(reference, changed, 'extra')
        self.assertEqual(len(rows), 4)
        self.assertTrue(any(not r['shape_matches'] for r in rows))
        # Numerical equality treats signed zero as equal; track byte equality separately.
        positive = {"trace['00_equilibrium']['zero']": np.array([0.])}
        negative = {"trace['00_equilibrium']['zero']": np.array([-0.])}
        with redirect_stdout(io.StringIO()):
            rows = check.compare(positive, negative, 'signed_zero')
        self.assertEqual(check.first_changed_stage(rows), 'none')
        self.assertEqual(check.first_changed_stage(rows, bitwise=True), '00_equilibrium')

    def test_reused_state_loader_accepts_both_saved_formats_and_rejects_invalid_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'state.npz'
            for prefix in ('', 'implicit__'):
                np.savez(path, **{prefix+'R_cos': np.array([[1.,2.]])})
                np.testing.assert_array_equal(check.load_named_state(path, ('R_cos',))['R_cos'], [[1.,2.]])
            with self.assertRaisesRegex(ValueError, 'required implicit fields'):
                check.load_named_state(path, ('Z_sin',))
            np.savez(path, implicit__R_cos=np.array([np.inf]))
            with self.assertRaises(FloatingPointError):
                check.load_named_state(path, ('R_cos',))

    def test_trace_parent_repeats_identical_configuration_and_forwards_frozen_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'trace'
            state = Path(temporary)/'state.npz'
            np.savez(state, implicit__R_cos=np.ones(1))
            def complete(command, **kwargs):
                folder = Path(command[command.index('--out-dir')+1])
                self.assertEqual(command[command.index('--worker')+1], 'fd')
                self.assertIn('--trace-inputs', command)
                self.assertIn('--trace-projections', command)
                self.assertEqual(command[command.index('--reuse-states')+1], str(state))
                np.savez(folder/'normalization_inputs_saved.npz', **{"trace['00_equilibrium']['R_cos']": np.ones(1)})
                (folder/'metadata.json').write_text(json.dumps(dict(config_sha256='config', input_sha256='input',
                    reused_state_sha256='state', versions={}, sources={})))
                return SimpleNamespace(returncode=0)
            with patch.object(sys, 'argv', [str(check.__file__), '--out-dir', str(output),
                    '--trace-inputs', '--trace-projections', '--reuse-states', str(state)]), \
                 patch.object(check.subprocess, 'run', side_effect=complete) as run, redirect_stdout(io.StringIO()) as log:
                check.main()
            self.assertEqual(run.call_count, 3)
            self.assertTrue((output/'baseline_3').is_dir())
            self.assertIn('first_changed_stage=none', log.getvalue())
            self.assertNotIn('jax', sys.modules)

    def test_comparison_separates_tables_inputs_and_nonfinite_masks(self):
        reference = {"support['geometry'].G_PS": np.array([np.nan, 1.]),
            "support['database'].D13": np.array([1.95343806242373]),
            "scan_inputs['channels'].rho": np.array([.75, .875]),
            "scan_inputs['source']": np.asarray("same-input")}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"snapshot.npz"
            np.savez(path, **reference, **{"explicit_support_bars[0]['database'].D13": np.ones(1)})
            loaded = check.saved_arrays(path)
        self.assertEqual(set(loaded), set(reference))
        with redirect_stdout(io.StringIO()):
            rows = check.compare(reference, loaded, "same")
        self.assertTrue(all(r["arrays_match"] and r["max_abs_difference"] == 0 for r in rows))
        changed = {k: v.copy() for k, v in reference.items()}
        changed["support['database'].D13"][0] = 1.9559692059134934
        changed["support['geometry'].G_PS"][0] = 0.
        del changed["scan_inputs['channels'].rho"]
        with redirect_stdout(io.StringIO()):
            rows = {r["leaf"]: r for r in check.compare(reference, changed, "changed")}
        self.assertAlmostEqual(rows["support['database'].D13"]["max_abs_difference"],
                               .0025311434897634655, places=15)
        self.assertTrue(np.isinf(rows["support['geometry'].G_PS"]["max_abs_difference"]))
        self.assertFalse(rows["support['geometry'].G_PS"]["nonfinite_values_match"])
        self.assertFalse(rows["scan_inputs['channels'].rho"]["shape_matches"])

    def test_generic_routes_change_only_scan_recording_and_skip_initialization(self):
        original = {"general": {"mode": "transport"}, "geometry": {"vmec_lane": "ad"},
            "neoclassical": {"flux_model": "ntx_scan_runtime"}}
        ad = check.baseline_config(original, "ad")
        fd = check.baseline_config(original, "fd")
        self.assertEqual(original["general"]["mode"], "transport")
        self.assertEqual(ad["geometry"], fd["geometry"])
        self.assertEqual(ad["general"]["mode"], "ambipolarity")
        self.assertEqual(fd["general"]["mode"], "ambipolarity")
        self.assertEqual(ad["neoclassical"]["ntx_scan_coefficient_reverse_mode"], "structured")
        self.assertTrue(ad["neoclassical"]["ntx_scan_record_primal"])
        self.assertEqual(fd["neoclassical"], original["neoclassical"])
        original["neoclassical"]["interpolation_mode"] = "preprocessed_3d_radial_ntss1d"
        with self.assertRaisesRegex(ValueError, "GENERIC"):
            check.baseline_config(original, "fd")

    def test_worker_refuses_cpu_before_importing_physics(self):
        with patch.dict(sys.modules, jax=SimpleNamespace(default_backend=lambda: "cpu")):
            with self.assertRaisesRegex(RuntimeError, "no CPU fallback"):
                check.worker(SimpleNamespace())

    def test_parent_runs_three_separate_workers_and_compares_their_saved_tables(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/"baselines"

            def complete(command, **kwargs):
                folder = Path(command[command.index("--out-dir")+1])
                offset = {"ad": 0., "fd_1": .01, "fd_2": .02}[folder.name]
                arrays = {key: np.array([1.+offset]) for key in check.TABLES}
                np.savez(folder/"generic_database_baseline_saved.npz", **arrays)
                return SimpleNamespace(returncode=0)

            with patch.object(sys, "argv", [str(check.__file__), "--out-dir", str(output)]), \
                 patch.object(check.subprocess, "run", side_effect=complete) as run, \
                 redirect_stdout(io.StringIO()):
                check.main()
            self.assertEqual(run.call_count, 3)
            routes = [call.args[0][-1] for call in run.call_args_list]
            self.assertEqual(routes, ["ad", "fd", "fd"])
            with (output/"comparison.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 9)
            self.assertEqual({r["comparison"] for r in rows},
                {"fresh_ad_vs_fd_1", "fresh_ad_vs_fd_2", "fd_1_vs_fd_2"})
            self.assertTrue(all("e" in r["max_abs_difference"] for r in rows))
            self.assertNotIn("jax", sys.modules)


if __name__ == "__main__":
    unittest.main()
