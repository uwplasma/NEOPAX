"""Saved-input validation for the point replay; no JAX or physical solves."""
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from contextlib import redirect_stdout
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

FILE = Path(__file__).resolve().parents[1]/'examples/benchmarks/check_saved_ntx_point_repeatability.py'
spec = importlib.util.spec_from_file_location('point_replay', FILE)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class PointReplayTests(unittest.TestCase):
    def test_geometry_capture_keeps_actual_calls_and_saves_boundaries_after_failure(self):
        active, calls = [], []
        state = SimpleNamespace(R_cos=np.asarray([[1.,2.]]))
        basis = np.eye(2)
        tables = SimpleNamespace(bmnc=np.asarray([[3.,4.]]), nfp=2)
        result = {'bmnc_b':np.asarray([[5.,6.]]), 'gmnc_b':np.asarray([[7.,8.]])}
        geometry = SimpleNamespace(B0=np.asarray([5.]))
        compiler = object()
        api = SimpleNamespace()
        geo = SimpleNamespace(_import_booz_xform_jax_api=lambda:api)
        def project(values, actual_basis):
            self.assertIs(values,state.R_cos)
            self.assertIs(actual_basis,basis)
            self.assertTrue(active[-1].endswith('_projection_compiler'))
            calls.append('project')
            return values@actual_basis
        def inputs(**kwargs):
            self.assertIs(kwargs['state'],state)
            self.assertTrue(active[-1].endswith('_boozer_inputs_compiler'))
            calls.append('inputs')
            geo._vmec_boozer_fourier_projection(state.R_cos,basis)
            return tables
        def boozer(**kwargs):
            self.assertIs(kwargs['inputs'],tables)
            self.assertIs(kwargs['jit'],True)
            self.assertTrue(active[-1].endswith('_boozer_compiler'))
            calls.append('boozer')
            return result
        def build(context, actual_state, **kwargs):
            self.assertIs(actual_state,state)
            self.assertEqual(kwargs,{'n_r':51})
            calls.append('geometry')
            actual_tables = geo._booz_xform_inputs_from_state(state=state,static=context)
            self.assertIs(api.booz_xform_from_inputs(inputs=actual_tables,jit=True),result)
            return geometry
        geo._build_neopax_geometry_from_state=build
        geo._booz_xform_inputs_from_state=inputs
        geo._vmec_boozer_fourier_projection=project
        api.booz_xform_from_inputs=boozer
        @replay.contextmanager
        def ir(directory, actual_compiler):
            self.assertIs(actual_compiler,compiler)
            active.append(directory.name)
            try:
                yield
            finally:
                active.pop()
        modules={'jax':SimpleNamespace(), 'jax._src':SimpleNamespace(compiler=compiler)}
        with tempfile.TemporaryDirectory() as directory,patch.dict(sys.modules,modules), \
             patch.object(sys,'path',[str(FILE.parent),*sys.path]), \
             patch.object(replay,'capture_ntx_ir',side_effect=ir),redirect_stdout(io.StringIO()):
            out=Path(directory)/'geometry'
            with self.assertRaisesRegex(ValueError,'original guard failure'):
                with replay.capture_geometry_boundaries(out,geo):
                    self.assertIs(geo._build_neopax_geometry_from_state(object(),state,n_r=51),geometry)
                    self.assertEqual(list(out.glob('*.npz')),[])
                    raise ValueError('original guard failure')
            manifest=json.loads((out/'boundaries.json').read_text())
            self.assertEqual([r['stage'] for r in manifest],['geometry','boozer_inputs','projection','boozer'])
            self.assertTrue(all(r['completed'] and r['fields'] for r in manifest))
            with np.load(out/manifest[1]['file'],allow_pickle=False) as z:
                np.testing.assert_array_equal(z['capture.state.R_cos'],state.R_cos)
                np.testing.assert_array_equal(z['capture.outputs.bmnc'],tables.bmnc)
            with np.load(out/manifest[3]['file'],allow_pickle=False) as z:
                np.testing.assert_array_equal(z['capture.inputs.bmnc'],tables.bmnc)
                np.testing.assert_array_equal(z['capture.outputs.bmnc_b'],result['bmnc_b'])
            self.assertIs(geo._build_neopax_geometry_from_state,build)
            self.assertIs(geo._booz_xform_inputs_from_state,inputs)
            self.assertIs(geo._vmec_boozer_fourier_projection,project)
            self.assertIs(api.booz_xform_from_inputs,boozer)
        self.assertEqual(calls,['geometry','inputs','project','boozer'])
        self.assertEqual(active,[])
        self.assertNotIn('jax',sys.modules)

    def test_live_capture_keeps_original_arguments_results_and_restores_both_aliases_on_failure(self):
        tree=SimpleNamespace(tree_flatten_with_path=lambda x:([('.matrix',x.matrix)],None)
            if hasattr(x,'matrix') else ([('',x)],None), keystr=lambda path:path)
        modules={'jax':SimpleNamespace(tree_util=tree,device_get=lambda x:x)}
        calls=[]
        prepared=SimpleNamespace(matrix=np.ones((2,2)))
        nu,es=np.asarray([.1,.1]),np.asarray([.2,.3])
        coefficients=np.arange(10,dtype=float).reshape(2,5)
        def scan(p,n,e):
            self.assertIs(p,prepared)
            self.assertIs(n,nu)
            self.assertIs(e,es)
            calls.append((p,n,e))
            return coefficients
        for owner in ('ntx._solver_scan','ntx._neopax_ad'):
            modules[owner]=SimpleNamespace(_scan_coefficients_serial=scan)
        with tempfile.TemporaryDirectory() as directory,patch.dict(sys.modules,modules):
            root=Path(directory)
            reference=root/'reference.npz'
            # A mismatch is evidence, not permission to replace an input or
            # mask the original build error.
            np.savez(reference,**{'prepared.matrix':np.zeros((2,2))})
            out=root/'capture'
            with redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,'original build failure'):
                with replay.capture_production_scans(out,reference=reference,point=(1,0,1)):
                    for owner in ('ntx._solver_scan','ntx._neopax_ad'):
                        actual=modules[owner]._scan_coefficients_serial(prepared,nu,es)
                        self.assertIs(actual,coefficients)
                        self.assertFalse(out.exists())
                    raise RuntimeError('original build failure')
            self.assertEqual(len(calls),2)
            for owner in ('ntx._solver_scan','ntx._neopax_ad'):
                self.assertIs(modules[owner]._scan_coefficients_serial,scan)
            for index in range(2):
                with np.load(out/f'surface_{index:04d}.npz',allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['prepared.matrix'],prepared.matrix)
                    np.testing.assert_array_equal(z['nu'],nu)
                    np.testing.assert_array_equal(z['Es'],es)
                    np.testing.assert_array_equal(z['coefficients'],coefficients)
            self.assertTrue((out/'preparation_comparison.csv').exists())
        self.assertNotIn('jax',sys.modules)

    def test_prepare_only_parent_uses_four_fresh_contexts_and_rejects_remaining_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for drift in (False,True):
                out=root/str(drift)
                commands=[]
                class Process:
                    def __init__(self, command, **kwargs):
                        commands.append(command)
                        folder=Path(command[command.index('--out-dir')+1])
                        mode=command[command.index('--worker')+1]
                        folder.mkdir(parents=True)
                        value=np.ones(2)
                        if drift and mode=='production_1':
                            value[0]=np.nextafter(1.,2.)
                        np.savez(folder/'fresh_prepared.npz',**{'prepared.geometry.b':value})
                        self.stdout=iter(['preparation saved\n'])
                    def __enter__(self): return self
                    def __exit__(self,*args): pass
                    def wait(self): return 0
                argv=['replay','--fd-run',str(root),'--input-reference',str(root),
                    '--ad-dir',str(root),'--source-metadata',str(root/'metadata.json'),
                    '--out-dir',str(out),'--prepare-only','--fd-parameter','density_shape_alpha',
                    '--input-pattern','geometry_build_*.npz','--capture-compiler-ir']
                with patch.object(replay,'preflight',return_value=None),patch.object(replay.subprocess,'Popen',Process), \
                     patch.object(sys,'argv',argv),redirect_stdout(io.StringIO()):
                    if drift:
                        with self.assertRaisesRegex(ValueError,'preparation still differs'):
                            replay.main()
                    else:
                        replay.main()
                self.assertEqual(len(commands),4)
                for command,layout in zip(commands,('point','point','production','production'),strict=True):
                    self.assertIn('--prepare-only',command)
                    self.assertNotIn('--frozen-prepared',command)
                    self.assertEqual(command[command.index('--scan-layout')+1],layout)
                    self.assertEqual(command[command.index('--fd-parameter')+1],'density_shape_alpha')
                    self.assertEqual(command[command.index('--input-pattern')+1],'geometry_build_*.npz')
                    self.assertIn('--capture-compiler-ir',command)
                self.assertTrue((out/'preparation_comparison.csv').is_file())
        self.assertNotIn('jax',sys.modules)

    def test_live_ir_capture_observes_actual_prepare_and_scan_calls_and_restores_on_failure(self):
        active, observed = [], []
        prepared = SimpleNamespace(matrix=np.ones((2,2)))
        surface, grid = object(), object()
        nu, es = np.ones(1), np.zeros(1)
        coefficients = np.ones((1,5))
        def prepare(s, g):
            self.assertIs(s,surface)
            self.assertIs(g,grid)
            self.assertTrue(active[-1].startswith('preparation_'))
            observed.append('prepare')
            return prepared
        def scan(p, n, e):
            self.assertIs(p,prepared)
            self.assertIs(n,nu)
            self.assertIs(e,es)
            self.assertEqual(active[-1],'surface_0000_compiler')
            observed.append('scan')
            return coefficients
        @replay.contextmanager
        def ir(directory, compiler):
            active.append(directory.name)
            try:
                yield
            finally:
                active.pop()
        modules = {name:SimpleNamespace(prepare_monoenergetic_system=prepare,
                    _scan_coefficients_serial=scan) for name in ('ntx','ntx._solver_scan','ntx._neopax_ad')}
        modules.update({'jax':SimpleNamespace(device_get=lambda x:x,tree_util=SimpleNamespace(
            tree_flatten_with_path=lambda x:([('.matrix',x.matrix)],'matrix') if hasattr(x,'matrix') else ([('',x)],'array'),
            keystr=lambda p:p)), 'jax._src':SimpleNamespace(compiler=object())})
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules,modules), \
             patch.object(replay,'capture_ntx_ir',side_effect=ir), redirect_stdout(io.StringIO()):
            out = Path(directory)/'capture'
            with self.assertRaisesRegex(ValueError,'original failure'):
                with replay.capture_production_scans(out,compiler_ir=True):
                    for name in ('ntx','ntx._solver_scan','ntx._neopax_ad'):
                        self.assertIs(modules[name].prepare_monoenergetic_system(surface,grid),prepared)
                    self.assertIs(modules['ntx._solver_scan']._scan_coefficients_serial(prepared,nu,es),coefficients)
                    raise ValueError('original failure')
            for name in ('ntx','ntx._solver_scan','ntx._neopax_ad'):
                self.assertIs(modules[name].prepare_monoenergetic_system,prepare)
                self.assertIs(modules[name]._scan_coefficients_serial,scan)
            self.assertTrue((out/'surface_0000.npz').exists())
            self.assertIn('prepared.__treedef__',(out/'surface_0000.txt').read_text())
        self.assertEqual(observed,['prepare','prepare','prepare','scan'])
        self.assertEqual(active,[])
        self.assertNotIn('jax',sys.modules)

    def test_worker_replays_saved_preparation_through_flat_arguments_without_global_options(self):
        class Ready(np.ndarray):
            def block_until_ready(self):
                return self
        compiled_options = []
        class Executable:
            def __init__(self, function):
                self.function = function
            def __call__(self, *args):
                return self.function(*args).view(Ready)
            def as_text(self):
                return 'mock compiled scan'
        def jit(function, *, compiler_options):
            compiled_options.append(compiler_options)
            return SimpleNamespace(lower=lambda *args:SimpleNamespace(compile=lambda:Executable(function)))
        tree = SimpleNamespace(tree_flatten_with_path=lambda p:([('.matrix',p.matrix)], 'prepared'),
            keystr=lambda path:path, tree_unflatten=lambda shape,values:SimpleNamespace(matrix=values[0]))
        def scan(prepared, nu, es):
            np.testing.assert_array_equal(prepared.matrix, [[1.]])
            if nu.size==1:
                np.testing.assert_array_equal(nu,[.1])
                np.testing.assert_array_equal(es,[.2])
                return np.asarray([[2.,0.,3.,5.,0.]])
            np.testing.assert_array_equal(nu,[.1,.1,.1,.2,.2,.2])
            np.testing.assert_array_equal(es,[.2,.3,.4,.2,.3,.4])
            return np.column_stack((np.ones(6)*2,np.zeros(6),3+np.arange(6),np.ones(6)*5,np.zeros(6)))
        prepare = unittest.mock.Mock(return_value=SimpleNamespace(matrix=np.ones((1,1))))
        modules = {'jax':SimpleNamespace(default_backend=lambda:'gpu',config=SimpleNamespace(x64_enabled=True),
                    tree_util=tree,device_get=lambda x:x,jit=jit), 'jax.numpy':np,
            'ntx.geometry':SimpleNamespace(VmecSurface=lambda **values:SimpleNamespace(**values)),
            'ntx':SimpleNamespace(GridSpec=lambda **values:SimpleNamespace(**values),prepare_monoenergetic_system=prepare),
            'ntx._solver_scan':SimpleNamespace(_scan_coefficients_sequential_impl=scan,
                _scan_coefficients_serial=unittest.mock.Mock(side_effect=lambda *args:scan(*args).view(Ready))),
            'ntx._geometry':SimpleNamespace(__file__=str(FILE),_sum_fourier_modes=lambda value:value),
            'NEOPAX._transport_flux_models':SimpleNamespace(_build_ntx_field_channels=lambda *args:(None,np.asarray([[.2,.3,.4]]),None))}
        # import jax.numpy also resolves numpy as an attribute of the fake parent.
        modules['jax'].numpy = np
        config = {'neoclassical':{'ntx_scan_n_theta':25,'ntx_scan_n_zeta':31,'ntx_scan_n_xi':64,
            'ntx_scan_rho':[.5],'ntx_scan_er_tilde':[.001,.002,.003],'ntx_scan_nu_v':[.1,.2]}}
        reference = {"trace['03_surfaces'][0]['path']":np.asarray('input.vmec'),
            "trace['03_surfaces'][0]['b_cos']":np.ones(1)}
        expected = {'support[\'database\'].'+name:np.ones((1,1,1)) for name in ('D11_log','D13','D33')}
        inputs = {"channels['drds']":np.asarray([2.])}
        before = dict(replay.os.environ)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules), redirect_stdout(io.StringIO()):
            root = Path(directory)
            for mode in ('native_1','native_2','controlled_1'):
                folder = root/mode
                folder.mkdir()
                args = SimpleNamespace(point=(0,0,0),out_dir=folder,worker=mode,
                    frozen_prepared=None if mode=='native_1' else root/'native_1',scan_layout='point',
                    repeats=2,prepare_only=False,live_scan=None,scan_kernel='flat')
                replay.worker(args,(config,expected,reference,expected,inputs))
                with np.load(folder/'result.npz') as result:
                    np.testing.assert_array_equal(result['table_0'],[np.log10(8.),6.,.5])
                    np.testing.assert_array_equal(result['coefficients_0'],result['coefficients_1'])
                # Later workers must retain this fresh drift as evidence but
                # use the first worker's exact arrays for the coefficient solve.
                prepare.return_value = SimpleNamespace(matrix=np.asarray([[9.]]))
            with np.load(root/'native_2/fresh_prepared.npz') as z:
                np.testing.assert_array_equal(z['prepared.matrix'],[[9.]])
            folder = root/'production'
            folder.mkdir()
            args = SimpleNamespace(point=(0,1,2),out_dir=folder,worker='native_1',
                frozen_prepared=root/'native_1',scan_layout='production',repeats=1,
                prepare_only=False,live_scan=None,scan_kernel='flat')
            expected = {key:np.ones((1,2,3)) for key in expected}
            replay.worker(args,(config,expected,reference,expected,inputs))
            with np.load(folder/'result.npz') as z:
                self.assertEqual(int(z['selected']),5)
                self.assertEqual(float(z['coefficients_0'][2]),8.)
                self.assertEqual(float(z['table_0'][1]),16.)
                self.assertEqual(z['scan_coefficients_0'].shape,(6,5))
                self.assertNotIn('coefficients_1',z.files)
            folder = root/'preparation_only'
            folder.mkdir()
            args = SimpleNamespace(point=(0,1,2),out_dir=folder,worker='native_1',
                frozen_prepared=None,scan_layout='production',repeats=1,
                prepare_only=True,live_scan=None,scan_kernel='flat')
            replay.worker(args,(config,expected,reference,expected,inputs))
            self.assertTrue((folder/'fresh_prepared.npz').is_file())
            self.assertFalse((folder/'compiled_hlo.txt').exists())
            self.assertFalse((folder/'result.npz').exists())
            # Restore the actual scan, even if the newly prepared matrix
            # differs. Call the production entry directly: no extra jit.
            live=root/'live.npz'
            live_nu=np.asarray([.1,.1,.1,.2,.2,.2])
            live_es=np.asarray([.2,.3,.4,.2,.3,.4])
            live_coeff=scan(SimpleNamespace(matrix=np.ones((1,1))),live_nu,live_es)
            np.savez(live,**{'prepared.matrix':np.ones((1,1)),
                'nu':live_nu,'Es':live_es,'coefficients':live_coeff})
            for kernel in ('production','flat'):
                folder=root/('live_'+kernel)
                folder.mkdir()
                args=SimpleNamespace(point=(0,1,2),out_dir=folder,worker='native_1',
                    frozen_prepared=None,scan_layout='point',repeats=2,
                    prepare_only=False,live_scan=live,scan_kernel=kernel)
                replay.worker(args,(config,expected,reference,expected,inputs))
                with np.load(folder/'result.npz',allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['scan_coefficients_0'],live_coeff)
                    np.testing.assert_array_equal(z['expected_live_coefficients'],live_coeff)
                    self.assertEqual(int(z['selected']),5)
                self.assertEqual((folder/'compiled_hlo.txt').exists(),kernel=='flat')
            self.assertEqual(modules['ntx._solver_scan']._scan_coefficients_serial.call_count,2)
            with self.assertRaisesRegex(ValueError,'shape or prepared'):
                replay.read_live_scan(live,(1,1,1),(0,0,0))
        self.assertEqual(compiled_options,[{},{},{'xla_gpu_deterministic_ops':True},{},{}])
        self.assertEqual(before,dict(replay.os.environ))
        self.assertNotIn('jax',sys.modules)

    def test_exact_input_alignment_detects_ulp_signed_zero_dtype_and_missing_fields(self):
        a = replay.scan_inputs({"trace['03_channels']['b00']":np.asarray([0.,2.]),
            "trace['04_surfaces'][0]['b_cos']":np.asarray([3.]), 'irrelevant':np.zeros(1)})
        b = replay.scan_inputs({"scan_inputs['channels']['b00']":np.asarray([0.,2.]),
            "scan_inputs['surfaces'][0].b_cos":np.asarray([3.])})
        self.assertEqual(replay.changes(a,b), [])
        for changed in (np.asarray([-0.,2.]), np.asarray([0.,np.nextafter(2.,3.)]),
                        np.asarray([0.,2.],dtype=np.float32), np.asarray([0.])):
            self.assertEqual(replay.changes(a,dict(b,**{"channels['b00']":changed})), ["channels['b00']"])
        self.assertEqual(replay.changes(a,{"channels['b00']":b["channels['b00']"]}), ["surfaces[0]['b_cos']"])

    def test_preflight_rejects_changed_inputs_geometry_and_sources_before_jax(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'fd_ZBS_1_0/snapshots').mkdir(parents=True)
            (root/'config.toml').write_text('[neoclassical]\n')
            args = SimpleNamespace(fd_run=root,input_reference=root,ad_dir=root,
                source_metadata=root/'metadata.json',point=(0,0,0),live_scan=None)
            source = root/'source.py'
            source.write_text('pass\n')
            args.source_metadata.write_text(json.dumps({'versions':{},'sources':{'ntx._solver':{
                'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}}}))
            geo = {"support['geometry'].a_b":np.asarray(1.)}
            fd = dict(geo, **{"scan_inputs['channels']['b00']":np.ones(1),
                "scan_inputs['surfaces'][0].b_cos":np.ones(1),
                "support['database'].D13":np.ones((1,1,1))})
            np.savez(root/'normalization_inputs_1.npz',**{
                "trace['02_channels']['b00']":np.ones(1),
                "trace['03_surfaces'][0]['b_cos']":np.ones(1)})
            np.savez(root/'ad_terminal_1.npz',**geo)
            path = root/'fd_ZBS_1_0/snapshots/fd_reference_support_1.npz'
            np.savez(path,**fd)
            self.assertEqual(len(replay.preflight(args)[-1]),2)
            fd["scan_inputs['channels']['b00']"][0] = np.nextafter(1.,2.)
            np.savez(path,**fd)
            with self.assertRaisesRegex(ValueError,'input fields differ'):
                replay.preflight(args)
            fd["scan_inputs['channels']['b00']"][0] = 1.
            fd["support['geometry'].a_b"] = np.asarray(2.)
            np.savez(path,**fd)
            with self.assertRaisesRegex(ValueError,'geometries differ'):
                replay.preflight(args)
            fd.update(geo)
            np.savez(path,**fd)
            source.write_text('changed\n')
            with self.assertRaisesRegex(ValueError,'Source changed'):
                replay.preflight(args)
            source.write_text('pass\n')
            args.point = (1,0,0)
            with self.assertRaisesRegex(ValueError,'outside the saved table'):
                replay.preflight(args)
            args.point = (0,0,0)
            # The current failed profile build and original-entry geometry
            # capture have different folder/pattern and source metadata forms.
            (root/'fd_density_shape_alpha/snapshots').mkdir(parents=True)
            path.rename(root/'fd_density_shape_alpha/snapshots'/path.name)
            (root/'normalization_inputs_1.npz').rename(root/'geometry_build_1.npz')
            args.fd_parameter = 'density_shape_alpha'
            args.input_pattern = 'geometry_build_*.npz'
            args.source_metadata.write_text(json.dumps({'versions':{'jax':'0.10.0'},
                'sources':{str(source):hashlib.sha256(source.read_bytes()).hexdigest()}}))
            (root/'ad/forward').mkdir(parents=True)
            (root/'ad/forward/metadata.json').write_text(json.dumps({'versions':{
                'jax':'0.10.0','ntx':'0.3.0','solvax':'0.36.2'}}))
            with patch.object(replay.importlib.metadata,'version',side_effect=lambda name:{
                    'jax':'0.10.0','ntx':'0.3.0','solvax':'0.36.2'}[name]) as version:
                self.assertEqual(len(replay.preflight(args)[-1]),2)
                self.assertEqual({call.args[0] for call in version.call_args_list},{'jax','ntx','solvax'})
                source.write_text('changed\n')
                with self.assertRaisesRegex(ValueError,'Source changed'):
                    replay.preflight(args)
            with patch.object(replay.importlib.metadata,'version',return_value='0.7.2'):
                with self.assertRaisesRegex(ValueError,'Package version changed'):
                    replay.preflight(args)
            (root/'ad/forward/metadata.json').write_text(json.dumps({'versions':{'jax':'0.7.2'}}))
            with self.assertRaisesRegex(ValueError,'Conflicting saved package versions'):
                replay.preflight(args)
        self.assertNotIn('jax',sys.modules)

    def test_compiler_capture_keeps_arguments_options_result_and_restores_on_failure(self):
        computation = SimpleNamespace(operation=SimpleNamespace(attributes={'sym_name':'jit__scan_coefficients_serial'}))
        executable = SimpleNamespace(hlo_modules=lambda:[SimpleNamespace(to_string=lambda:'actual executable')])
        compile_call = unittest.mock.Mock(return_value=executable)
        compiler = SimpleNamespace(compile_or_get_cached=compile_call)
        backend, options = object(), object()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/'compiler'
            with self.assertRaisesRegex(RuntimeError,'original worker failure'):
                with replay.capture_ntx_ir(out,compiler):
                    actual = compiler.compile_or_get_cached(backend,computation,compile_options=options)
                    self.assertIs(actual,executable)
                    compile_call.assert_called_once_with(backend,computation,compile_options=options)
                    raise RuntimeError('original worker failure')
            self.assertIs(compiler.compile_or_get_cached,compile_call)
            programs = json.loads((out/'programs.json').read_text())
            self.assertEqual(programs[0]['name'],'jit__scan_coefficients_serial')
            self.assertEqual((out/'program_0000.txt').read_text(),'actual executable')
            self.assertEqual((out/'program_0000.mlir').read_text(),str(computation))
        self.assertNotIn('jax',sys.modules)

    def test_table_projection_uses_the_production_factors_and_d11_floor(self):
        np.testing.assert_array_equal(replay.normalized(np.asarray([2.,0.,3.,5.,0.]),2.,.1),
                                      [np.log10(8.),6.,.5])
        self.assertEqual(replay.normalized(np.asarray([-2.,0.,3.,5.,0.]),2.,.1)[0],-20.)


if __name__ == '__main__':
    unittest.main()
