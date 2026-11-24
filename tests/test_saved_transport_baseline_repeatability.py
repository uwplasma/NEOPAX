"""Saved-array diagnostics only: these tests never import JAX or run physics."""
import argparse
import ast
import contextlib
import dataclasses
import importlib.util
import json
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "examples/benchmarks/check_saved_transport_baseline_repeatability.py"
spec = importlib.util.spec_from_file_location("baseline_repeat_check", SCRIPT)
check = importlib.util.module_from_spec(spec)
_jax_before = "jax" in sys.modules
spec.loader.exec_module(check)
_jax_after = "jax" in sys.modules


class SavedBaselineTests(unittest.TestCase):
    def test_compiler_observer_preserves_calls_results_and_restores_hook(self):
        class Computation:
            def __init__(self,name,text):
                self.operation=SimpleNamespace(attributes={'sym_name':name})
                self.text=text
            def __str__(self):
                return self.text
        options, backend = object(),object()
        calls,executables=[],[]
        def compile_program(*args,**kwargs):
            calls.append((args,kwargs))
            computation=args[1] if len(args)>1 else kwargs['computation']
            executable=SimpleNamespace(hlo_modules=lambda:[SimpleNamespace(to_string=lambda:'compiled '+str(computation))])
            executables.append(executable)
            return executable
        original=Mock(side_effect=compile_program)
        compiler=SimpleNamespace(compile_or_get_cached=original)
        modules={'jax._src':SimpleNamespace(compiler=compiler)}
        with tempfile.TemporaryDirectory() as directory,patch.dict(sys.modules,modules):
            out=Path(directory)
            with check.capture_forward_compiler_ir(out):
                result=compiler.compile_or_get_cached(backend,Computation('jit_add','ignored'),options)
                self.assertIs(result,executables[0])
                first=compiler.compile_or_get_cached(backend,Computation('jit_scan','small'),options)
                last=compiler.compile_or_get_cached(backend=backend,
                    computation=Computation('jit_scan','largest native program'),compile_options=options)
                self.assertEqual(first.hlo_modules()[0].to_string(),'compiled small')
                self.assertEqual(last.hlo_modules()[0].to_string(),'compiled largest native program')
                self.assertIs(first,executables[1])
                self.assertIs(last,executables[2])
            self.assertIs(compiler.compile_or_get_cached,original)
            self.assertEqual(original.call_count,3)
            self.assertIs(calls[0][0][2],options)
            self.assertIs(calls[2][1]['compile_options'],options)
            self.assertEqual((out/'adaptive_stablehlo.mlir').read_text(),'largest native program')
            self.assertEqual((out/'adaptive_compiled_hlo.txt').read_text(),'compiled largest native program')
            self.assertEqual(len(json.loads((out/'compiler_programs.json').read_text())),2)
            with self.assertRaisesRegex(RuntimeError,'original failure'):
                with check.capture_forward_compiler_ir(out):
                    raise RuntimeError('original failure')
            self.assertIs(compiler.compile_or_get_cached,original)
        self.assertEqual(_jax_before,'jax' in sys.modules)

    def test_raw_initial_state_roundtrip_preserves_zero_and_subfloor_helium(self):
        source = ast.parse((SCRIPT.parents[2]/'NEOPAX/_transport_solvers.py').read_text())
        functions = [n for n in source.body if isinstance(n,ast.FunctionDef)
                     and n.name in {'_electron_density_index','_pack_transport_state_arrays'}]
        namespace = dict(Any=object,dataclasses=dataclasses,jnp=np)
        exec(compile(ast.Module(body=functions,type_ignores=[]),'production_packing','exec'),namespace)
        State = dataclasses.make_dataclass('State',[('density',object),('pressure',object),('Er',object)])
        for names in (('e','D','T','He'),('D','e','He'),('D','He')):
            species = SimpleNamespace(names=names)
            density = np.arange(len(names)*3,dtype=float).reshape(len(names),3)+1.
            density[names.index('He')] = [0.,-0.,1e-12]
            pressure = density*2
            er = np.asarray([0.,-0.,-3.])
            original = State(density,pressure,er)
            packed = np.concatenate([a.reshape(-1) for a in namespace['_pack_transport_state_arrays'](original,species)])
            reference = {'final_state.Er':np.ones(3)*99.,
                         'final_state.pressure':np.ones_like(pressure)*99.}
            restored = State(**check.raw_initial_state_arrays(reference,packed,names))
            again = np.concatenate([a.reshape(-1) for a in namespace['_pack_transport_state_arrays'](restored,species)])
            self.assertEqual(packed.tobytes(),again.tobytes())
            self.assertEqual(restored.density[names.index('He')].tobytes(),density[names.index('He')].tobytes())
            with self.assertRaisesRegex(ValueError,'layout'):
                check.raw_initial_state_arrays(reference,packed[:-1],names)
        # Reproduce why the former output-unpacker route was not lossless:
        # passing None to the production density-floor helper means 1e-6.
        source = ast.parse((SCRIPT.parents[2]/'NEOPAX/_state.py').read_text())
        functions = [n for n in source.body if isinstance(n,ast.FunctionDef)
                     and n.name in {'_broadcast_species_floor','safe_density','apply_transport_density_floor'}]
        namespace = dict(Array=object,dataclasses=dataclasses,jnp=np,DEFAULT_TRANSPORT_DENSITY_FLOOR=1e-6)
        exec(compile(ast.Module(body=functions,type_ignores=[]),'production_density_floor','exec'),namespace)
        projected = namespace['apply_transport_density_floor'](restored,None)
        np.testing.assert_array_equal(projected.density[names.index('He')],[1e-6]*3)
        self.assertEqual(restored.density[names.index('He')].tobytes(),np.asarray([0.,-0.,1e-12]).tobytes())

    def test_first_attempt_difference_uses_time_order_and_exact_bits(self):
        reference = {name:np.ones(4) for name in check.SCHEDULE_FIELDS}
        current = {name:values.copy() for name, values in reference.items()}
        self.assertEqual(check.first_attempt_difference(reference, current), [])
        current['attempted_dts'][3] += .25
        current['err_norms'][1] = np.nextafter(1., np.inf)
        current['jacobian_reused'][1] = 0.
        rows = check.first_attempt_difference(reference, current)
        self.assertEqual({r['attempt'] for r in rows}, {2})
        self.assertEqual({r['field'] for r in rows}, {'err_norms','jacobian_reused'})
        reference['step_ts'][0], current['step_ts'][0] = 0., -0.
        self.assertEqual(check.first_attempt_difference(reference, current)[0]['attempt'], 1)
        current = {name:values[:2].copy() for name, values in reference.items()}
        self.assertEqual({r['attempt'] for r in check.first_attempt_difference(reference,current)}, {3})
        current['step_ts'] = np.zeros(2, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, 'step_ts'):
            check.first_attempt_difference(reference, current)

    def test_forward_trace_runs_only_sequential_saved_input_controls_and_checks_archives(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run, out = root/'saved', root/'trace'
            (run/'ad/snapshots').mkdir(parents=True)
            (run/'fd_n0/snapshots').mkdir(parents=True)
            endpoint = dict(final_y=np.asarray([2.]), final_time=np.asarray(.1))
            np.savez(run/'ad/snapshots/ad_terminal_a.npz', **endpoint)
            np.savez(run/'fd_n0/snapshots/fd_reference_baseline_b.npz', **endpoint)
            calls = []
            def launch(command, **kwargs):
                name = command[command.index('--variant')+1]
                folder = out/name
                calls.append(name)
                self.assertTrue(folder.is_dir())
                self.assertIn('--trace-ad-fd', command)
                self.assertIn('--capture-compiler-ir', command)
                self.assertEqual(Path(command[1]), SCRIPT)
                self.assertEqual(kwargs['env'], {'same_environment_as_paired_run':'1'})
                setup = {'initial_carry.y':np.asarray([1.]), 'initial_rhs':np.asarray([3.]),
                         'execution.reverse_direct_stage_adjoint':np.asarray(name.startswith('ad'))}
                if name == 'fd':
                    setup['initial_rhs'][0] = np.nextafter(3.,np.inf)
                np.savez(folder/'setup.npz', **setup)
                trajectory = {k:np.ones(3) for k in check.SCHEDULE_FIELDS}
                if name == 'ad':
                    trajectory['err_norms'][1] += .125
                if name == 'ad_no_checkpoints':
                    trajectory['next_dts'][2] += 1e-6
                np.savez(folder/'native.npz', **trajectory, **endpoint)
                return SimpleNamespace(returncode=0)
            argv = [str(SCRIPT),'--run-dir',str(run),'--out-dir',str(out),'--trace-ad-fd','--capture-compiler-ir']
            module = SimpleNamespace(worker_environment=lambda:{'same_environment_as_paired_run':'1'})
            with patch.dict(sys.modules, {'benchmark_transport_ntss1d_ad_fd':module}), \
                 patch.object(sys,'argv',argv), patch.object(check.subprocess,'run',side_effect=launch), \
                 contextlib.redirect_stdout(io.StringIO()):
                check.main()
            self.assertEqual(calls, list(check.FORWARD_VARIANTS))
            rows = json.loads((out/'comparison.json').read_text())
            self.assertEqual(rows[0]['initial'][0]['field'], 'initial_rhs')
            self.assertEqual(rows[0]['first_attempt'], [])
            self.assertEqual(rows[1]['initial'], [])
            self.assertTrue(rows[1]['context'])
            self.assertEqual(rows[1]['first_time_map'][0]['attempt'],3)
            self.assertEqual(rows[1]['first_time_map'][0]['reference_error'],'1.0')
            self.assertEqual(rows[2]['first_attempt'][0]['attempt'], 2)
            self.assertEqual(rows[-1]['differences'], [])
            self.assertEqual(rows[-2]['differences'], [])
            # Failed workers halt before subsequent controls or comparisons.
            calls.clear()
            argv[argv.index('--out-dir')+1] = str(root/'failed')
            def failed_worker(command, **kwargs):
                kwargs['stdout'].write('Traceback: original worker failure\n')
                return SimpleNamespace(returncode=1)
            stderr = io.StringIO()
            with patch.dict(sys.modules, {'benchmark_transport_ntss1d_ad_fd':module}),patch.object(sys,'argv',argv), \
                 patch.object(check.subprocess,'run',side_effect=failed_worker) as launch, \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr), \
                 self.assertRaises(SystemExit):
                check.main()
            self.assertEqual(launch.call_count, 1)
            self.assertIn('Traceback: original worker failure', stderr.getvalue())
        self.assertEqual(_jax_before, 'jax' in sys.modules)

    def test_forward_workers_preserve_independent_schedules_and_change_one_setup_boundary(self):
        @dataclasses.dataclass
        class Carry:
            y: object
            prev_stages: object
        @dataclasses.dataclass
        class Execution:
            reverse_direct_stage_adjoint: bool
        state, live, fixed = object(), SimpleNamespace(species=object()), SimpleNamespace(species=object())
        carry = Carry(np.asarray([1.,2.]),np.asarray([3.,4.,3.,4.]))
        prepared = SimpleNamespace(initial_carry=carry)
        equation = SimpleNamespace(fixed_temperature_profile=np.asarray([5.]))
        trajectory = SimpleNamespace(trace=SimpleNamespace(**{k:np.ones(3) for k in check.SCHEDULE_FIELDS}),
            final_carry=SimpleNamespace(y=np.asarray([2.,3.]),t=np.asarray(.1)),
            failed=False,completed=True,accepted_count=16,attempt_count=3)
        schedule_calls = []
        def adaptive(execution, carry_arg, **options):
            self.assertIs(carry_arg, carry)
            self.assertTrue(options['capture_newton_setup'])
            self.assertEqual(options['stop_after_accepted_steps'],16)
            schedule_calls.append((execution.reverse_direct_stage_adjoint,options.get('capture_segment_length')))
            return trajectory
        reverse = SimpleNamespace(initial_state_for_parameter_vector=Mock(side_effect=AssertionError('root evaluated')))
        reverse._radau_adaptive_schedule_rollout = adaptive
        reverse._reverse_adaptive_schedule_rollout = lambda execution, carry, **opts: adaptive(
            execution, carry, capture_newton_setup=opts.get('capture_segment_length') is not None,**opts)
        production = ast.parse((SCRIPT.parents[2]/'NEOPAX/_reverse_ad_transport.py').read_text())
        definition = next(n for n in production.body if isinstance(n,ast.FunctionDef)
                          and n.name=='prepare_reverse_static_setup')
        allowed = {a.arg for a in definition.args.kwonlyargs}
        def prepare(values, **options):
            self.assertTrue(set(options).issubset(allowed))
            self.assertIs(options['runtime'],fixed)
            self.assertIs(reverse.initial_state_for_parameter_vector(values),state)
            self.assertEqual(options['reverse_stage_adjoint_solve_mode'],'block')
            self.assertEqual(options['reverse_schedule_artifact_mode'],'reuse_static_probe')
            execution = Execution(True)
            options['schedule_probe_runner'](execution,carry,max_total_steps=256,
                stop_after_accepted_steps=16,capture_segment_length=4)
            # Setup can add reverse metadata after its probe. The snapshot
            # must retain the context actually passed to the forward call.
            return SimpleNamespace(prepared_rollout=prepared,execution_context=Execution(False),
                solve_vector_field=SimpleNamespace(__self__=equation))
        reverse.prepare_reverse_static_setup = prepare
        def fd_schedule(config, runtime, state_arg, **options):
            self.assertIn(runtime,(live,fixed))
            self.assertIs(state_arg,state)
            execution = Execution(False)
            adaptive(execution,carry,max_total_steps=256,stop_after_accepted_steps=options['accepted_step_limit'],
                capture_newton_setup=options['capture_newton_setup'])
            return {'equation_system':equation},prepared,execution,trajectory,state
        config = {'neoclassical':{'flux_model':'ntx_scan_runtime'}}
        orch = SimpleNamespace(_resolve_er_right_boundary_mode=lambda *a:'floating_ambipolar_edge',
            _normalize_solver_config=lambda c:{},_execution_device_context=lambda c:contextlib.nullcontext())
        modules = {
            'jax':SimpleNamespace(numpy=np,config=SimpleNamespace(x64_enabled=True),default_backend=lambda:'gpu',
                flatten_util=SimpleNamespace(ravel_pytree=lambda parts:(np.concatenate(parts),None))),
            'jax.numpy':np,
            'NEOPAX':SimpleNamespace(_orchestrator=orch,_reverse_ad_transport=reverse),
            'NEOPAX._transport_solvers':SimpleNamespace(_pack_transport_state_arrays=lambda *a:(carry.y,)),
            'NEOPAX._reverse_ad_initial_er':SimpleNamespace(runtime_with_fixed_ntx_database_model=lambda r:fixed),
            'benchmark_transport_realtime_geometry_forward_fd':SimpleNamespace(_schedule_rollout=fd_schedule),
            'benchmark_transport_autodiff_lagged_ntx':SimpleNamespace(
                _prepare_benchmark_config=lambda *a,**k:config,_baseline_profile_cfg=lambda c:{}),
            'benchmark_transport_reverse_ad_only':SimpleNamespace(PARAMETER_ORDER=('n0',),
                _profile_cfg_scalar_value=lambda *a:1.),
        }
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('ad/snapshots','fd_n0/snapshots'):
                (root/name).mkdir(parents=True)
            physical={'initial_y':carry.y,"support['geometry'].a_b":np.asarray(1.),
                      "support['database'].D13":np.asarray([2.])}
            np.savez(root/'ad/snapshots/ad_terminal_a.npz',**physical)
            np.savez(root/'fd_n0/snapshots/fd_reference_support_b.npz',**physical)
            np.savez(root/'fd_n0/snapshots/fd_reference_baseline_c.npz',initial_y=carry.y)
            (root/'ad/summary.json').write_text(json.dumps({'accepted_step_limit':16}))
            before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with patch.dict(sys.modules,modules),patch.object(check,'restore_runtime',return_value=(live,state)), \
                 patch.object(check,'version',return_value='test'),contextlib.redirect_stdout(io.StringIO()):
                for variant in check.FORWARD_VARIANTS:
                    out=root/variant
                    out.mkdir(exist_ok=True)
                    check.trace_forward_worker(root,out,variant,16)
                    np.testing.assert_array_equal(check.read_one(out,'setup.npz')['initial_rhs'],[3.,4.])
                    self.assertEqual(bool(check.read_one(out,'setup.npz')['execution.reverse_direct_stage_adjoint']),
                                     variant.startswith('ad'))
            self.assertEqual(schedule_calls,[(False,None),(False,None),(True,None),(True,4)])
            self.assertTrue(all(p.read_bytes()==value for p,value in before.items()))
        self.assertEqual(_jax_before,'jax' in sys.modules)

    def test_fd_device_context_matches_ad_for_default_and_explicit_placement(self):
        root = SCRIPT.parents[2]
        namespace = dict(Any=object, contextlib=contextlib, Path=Path,
            jax=SimpleNamespace(default_backend=lambda: 'gpu',
                local_devices=lambda backend: [backend+':0'],
                default_device=lambda device: contextlib.nullcontext(device)))
        for filename, names in (
            (root/'NEOPAX/_orchestrator.py', ('_normalized_general_device', '_execution_device_context')),
            (SCRIPT.with_name('benchmark_transport_reverse_ad_only.py'), ('_benchmark_device_context',)),
        ):
            functions = [n for n in ast.parse(filename.read_text()).body
                         if isinstance(n, ast.FunctionDef) and n.name in names]
            exec(compile(ast.Module(body=functions, type_ignores=[]), str(filename), 'exec'), namespace)
        source = SCRIPT.with_name('benchmark_transport_realtime_geometry_forward_fd.py')
        main = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'main')
        start = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
                     and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'device_arg')
        end = next(i for i, n in enumerate(main.body[start:], start) if isinstance(n, ast.Expr)
                   and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                   and isinstance(n.value.func.value, ast.Name)
                   and n.value.func.value.id == 'device_context')
        function = ast.parse('def placement(args):\n    pass\n').body[0]
        function.body = main.body[start:end] + ast.parse('return device_context, config').body
        namespace['parser'] = SimpleNamespace(error=lambda message: self.fail(message))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                     str(source), 'exec'), namespace)
        for configured in ('auto', 'cpu', 'gpu'):
            for requested in (None, 'default', 'auto', 'cpu', 'gpu'):
                with self.subTest(configured=configured, requested=requested):
                    def prepare(*, device, **kwargs):
                        return {'general': {'device': configured if device is None else device}}
                    namespace['_prepare_benchmark_config'] = Mock(side_effect=prepare)
                    args = SimpleNamespace(device=requested, config='unused.toml',
                        radau_jacobian_reuse_mode='legacy', geometry_fd_native_precision='float64',
                        geometry_input_only_dir=None, vmec_response_mode=None, ntx_scan_audit_dir=None)
                    context, config = namespace['placement'](args)
                    with context as fd, namespace['_benchmark_device_context'](config) as ad:
                        self.assertEqual(fd, ad)
                    expected = None if requested in (None, 'default') else requested
                    self.assertEqual(namespace['_prepare_benchmark_config'].call_args.kwargs['device'], expected)
        self.assertEqual(_jax_before, 'jax' in sys.modules)

    def test_saved_ad_geometry_reuses_exact_state_and_named_direction_without_solves(self):
        source = SCRIPT.with_name('benchmark_transport_realtime_geometry_forward_fd.py')
        names = ('_saved_ad_geometry_linearization','_frozen_linearized_vmec_geometry_bundle')
        functions = [n for n in ast.parse(source.read_text()).body
                     if isinstance(n,ast.FunctionDef) and n.name in names]
        State = dataclasses.make_dataclass('State',[('R_cos',object),('L_sin',object)])
        Params = dataclasses.make_dataclass('Params',[('boundary',object)])
        Raw = dataclasses.make_dataclass('Raw',[(n,object) for n in
            ('implicit','implicit_params','implicit_cfg','state','dof_mask','param_entries','vmec_response_mode')])
        params = Params(np.asarray([0.,2.]))
        state = State(np.asarray([3.,4.]),np.asarray([5.,6.]))
        mask = State(np.asarray([True,False]),np.asarray([True,True]))
        tangent = State(np.asarray([.1,.2]),np.asarray([.3,.4]))
        entry = dict(family='ZBS',m=1,n=0,baseline_coefficient=2.)
        context = SimpleNamespace(vmec_response_mode='raw_block',mboz=36,nboz=11)
        cfg,implicit = object(),object()
        construct = Mock(return_value=(implicit,params,cfg))
        raw = Raw(implicit,params,cfg,state,mask,(entry,),'raw_block')
        solve,linearize,record = Mock(return_value=raw),Mock(return_value=tangent),Mock()
        namespace = dict(np=np,jnp=np,Path=Path,Any=object,dataclasses=dataclasses,
            normalize_vmec_response_mode=lambda x:x,
            _geometry_context_from_config=lambda *a:context,
            _geometry_param_specs_from_parameter_name=lambda p:(('ZBS',1,0),),
            boundary_param_entries=lambda *a:(entry,),geometry_raw_block_solve_from_param_vector=solve,
            _manual_implicit_forward_state_tangent=linearize,_param_unit_tangent_like=lambda *a:object(),
            save_gradient_audit=record)
        exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),namespace)
        modules = {'NEOPAX._geometry_autodiff':SimpleNamespace(GeometryRawBlockSolve=Raw,
                _current_implicit_params_cfg_for_param_vector=construct),
            'vmex.core.solver':SimpleNamespace(SpectralState=State),
            'check_saved_transport_baseline_repeatability':check,
            'check_saved_compact_geometry_gradient':SimpleNamespace(named_arrays=lambda a,p:
                {k[len(p):]:v for k,v in a.items() if k.startswith(p) and '.' not in k[len(p):]})}
        config = dict(geometry=dict(vmec_lane='ad',n_radial=3))
        with tempfile.TemporaryDirectory() as directory,patch.dict(sys.modules,modules),contextlib.redirect_stdout(io.StringIO()):
            folder = Path(directory)
            capture = {'parameter_labels[0]':np.asarray('RBC:1:0'),
                'parameter_labels[1]':np.asarray('ZBS:1:0'),'vmec_response_mode':np.asarray('raw_block'),
                "payload_bars[0]['geometry'].r_grid":np.arange(3.),'params.boundary':params.boundary}
            dots = {'parameter_labels[0]':np.asarray('RBC:1:0'),'parameter_labels[1]':np.asarray('ZBS:1:0')}
            for f in dataclasses.fields(State):
                capture['state.'+f.name] = getattr(state,f.name)
                capture['mask.'+f.name] = getattr(mask,f.name)
                dots['state_tangents.'+f.name] = np.stack((np.ones(2)*7.,getattr(tangent,f.name)))
            def write(a=capture,b=dots):
                np.savez(folder/'ad_compact_geometry_one.npz',**a)
                np.savez(folder/'ad_compact_tangents_one.npz',**b)
            write()
            bundle = namespace['_frozen_linearized_vmec_geometry_bundle'](config,'ZBS:1:0',
                baseline_delta=0.,retain_raw_solve=True,ad_geometry_reference=folder)
            for f in dataclasses.fields(State):
                np.testing.assert_array_equal(getattr(bundle['state_star'],f.name),getattr(state,f.name))
                np.testing.assert_array_equal(getattr(bundle['state_tangent'],f.name),getattr(tangent,f.name))
            self.assertIs(bundle['raw_block_solve'].implicit_cfg,cfg)
            self.assertEqual(bundle['coefficient_value'],2.)
            solve.assert_not_called();linearize.assert_not_called()
            self.assertEqual(record.call_args.args,('fd_geometry_linearization',))
            self.assertEqual(record.call_args.kwargs['ad_geometry_reference'],str(folder.resolve()))
            construct.assert_called_once()
            # Reject incompatible references, including signed zero and dtype.
            for bad in (
                dict(capture,**{'params.boundary':np.asarray([1.,3.])}),
                dict(capture,**{'params.boundary':np.asarray([-0.,2.])}),
                dict(capture,**{'params.boundary':params.boundary.astype(np.float32)}),
                dict(capture,vmec_response_mode=np.asarray('main')),
                dict(capture,**{'state.R_cos':np.asarray([np.nan,4.])}),
                dict(capture,**{'mask.R_cos':np.asarray([True])}),
                dict(capture,**{"payload_bars[0]['geometry'].r_grid":np.arange(4.)}),
            ):
                write(bad)
                with self.assertRaises(ValueError):
                    namespace['_frozen_linearized_vmec_geometry_bundle'](config,'ZBS:1:0',baseline_delta=0.,ad_geometry_reference=folder)
            write(capture,dict(dots,**{'parameter_labels[1]':np.asarray('wrong')}))
            with self.assertRaisesRegex(ValueError,'labels'):
                namespace['_frozen_linearized_vmec_geometry_bundle'](config,'ZBS:1:0',baseline_delta=0.,ad_geometry_reference=folder)
            write(capture,dict(dots,**{'state_tangents.R_cos':np.ones((1,2))}))
            with self.assertRaisesRegex(ValueError,'linearization'):
                namespace['_frozen_linearized_vmec_geometry_bundle'](config,'ZBS:1:0',baseline_delta=0.,ad_geometry_reference=folder)
            with self.assertRaisesRegex(ValueError,'ad lane'):
                namespace['_frozen_linearized_vmec_geometry_bundle']({'geometry':{'vmec_lane':'forward'}},'ZBS:1:0',baseline_delta=0.,ad_geometry_reference=folder)
            solve.assert_not_called();linearize.assert_not_called()
            # No opt-in flag: retain the existing equilibrium and tangent calls.
            bundle = namespace['_frozen_linearized_vmec_geometry_bundle'](config,'ZBS:1:0',baseline_delta=0.)
            self.assertIs(bundle['state_star'],state)
            self.assertIs(bundle['state_tangent'],tangent)
            solve.assert_called_once();linearize.assert_called_once()

    def test_ad_reference_changes_only_er_and_requires_an_exact_forward_baseline(self):
        @dataclasses.dataclass
        class Geometry:
            scale: object
        @dataclasses.dataclass
        class Database:
            table: object
        geometry, database = Geometry(np.asarray(1.2)), Database(np.asarray([1., 2.]))
        reference = {"support['geometry'].scale":np.asarray(1.2),
                     "support['database'].table":np.asarray([1., 2.]),
                     'initial_y':np.asarray([1., 2., 3., 4., 5., 6.]),
                     'final_y':np.asarray([2., 3., 4., 5., 6., 7.]),
                     'final_time':np.asarray(1e-4)}
        initial = reference['initial_y'].copy()
        initial[-1] = np.nextafter(initial[-1], np.inf)
        kwargs = dict(geometry=geometry, database=database, initial_y=initial, er_size=2)
        er = check.ad_reference_initial_er(reference, **kwargs)
        np.testing.assert_array_equal(er, reference['initial_y'][-2:])
        er[0] = 100.
        self.assertEqual(reference['initial_y'][-2], 5.)
        diagnostic = io.StringIO()
        with contextlib.redirect_stdout(diagnostic), self.assertRaisesRegex(ValueError, 'geometry/database'):
            check.ad_reference_initial_er(reference, **dict(kwargs, database=Database(np.asarray([1.,3.]))))
        self.assertIn("support['database'].table",diagnostic.getvalue())
        self.assertIn("'max_abs': 1.0",diagnostic.getvalue())
        for bad in (
            dict(kwargs, database=None),
            dict(kwargs, database=Database(np.asarray([1., 3.]))),
            dict(kwargs, geometry=Geometry(np.asarray(1.2, dtype=np.float32))),
            dict(kwargs, initial_y=initial[:-1]),
            dict(kwargs, initial_y=np.asarray([9., 2., 3., 4., 5., 6.])),
            dict(kwargs, er_size=0),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                check.ad_reference_initial_er(reference, **bad)
        for bad in ({k:v for k, v in reference.items() if k != 'final_time'},
                    dict(reference, initial_y=np.asarray([1., 2., 3., 4., np.nan, 6.]))):
            with self.assertRaises(ValueError):
                check.ad_reference_initial_er(bad, **kwargs)
        current = {k:reference[k].copy() for k in ('initial_y', 'final_y', 'final_time')}
        with contextlib.redirect_stdout(io.StringIO()):
            check.require_ad_reference_baseline(reference, **current)
            for key in current:
                changed = {k:v.copy() for k, v in current.items()}
                changed[key].flat[0] = np.nextafter(changed[key].flat[0], np.inf)
                with self.subTest(boundary=key), self.assertRaisesRegex(ValueError, 'before FD endpoints'):
                    check.require_ad_reference_baseline(reference, **changed)

        # Check the actual benchmark wiring: validate before evolution, then
        # check its returned carry before objectives or endpoint construction.
        source = SCRIPT.with_name('benchmark_transport_realtime_geometry_forward_fd.py')
        main = next(node for node in ast.parse(source.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'main')
        calls = lambda node: {getattr(n.func, 'id', '') for n in ast.walk(node) if isinstance(n, ast.Call)}
        reference_setup = next(i for i, node in enumerate(main.body) if 'ad_reference_initial_er' in calls(node))
        rollout = next(i for i, node in enumerate(main.body) if '_schedule_rollout' in calls(node))
        guard = next(i for i, node in enumerate(main.body) if 'require_ad_reference_baseline' in calls(node))
        objective = next(i for i, node in enumerate(main.body) if '_objective_vector' in calls(node))
        self.assertLess(reference_setup, rollout)
        self.assertLess(rollout, guard)
        self.assertLess(guard, objective)

    def test_saved_runtime_preserves_interpolation_dispatch_and_all_database_leaves(self):
        class Monoenergetic:
            def __init__(self, **fields):
                self.__dict__.update(fields)
        class MonoenergeticRuntimeNTSS1D(Monoenergetic):
            pass
        @dataclasses.dataclass
        class Neo:
            database: object = None
        @dataclasses.dataclass
        class Flux:
            neoclassical_model: object

        # Exercise the real class-based dispatcher without importing JAX.
        source = SCRIPT.parents[2]/'NEOPAX/_monoenergetic.py'
        functions = [node for node in ast.parse(source.read_text()).body
                     if isinstance(node, ast.FunctionDef) and node.name in
                     ('normalize_interpolation_mode', 'monoenergetic_database_kind')]
        namespace = dict(Any=object, MONOENERGETIC_KIND_GENERIC='generic',
            MONOENERGETIC_KIND_BY_CLASS={MonoenergeticRuntimeNTSS1D:'preprocessed_3d_radial_ntss1d',
                                        Monoenergetic:'generic'})
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
        modules = {
            'NEOPAX._database': SimpleNamespace(Monoenergetic=Monoenergetic,
                MonoenergeticRuntimeNTSS1D=MonoenergeticRuntimeNTSS1D),
            'NEOPAX._monoenergetic': SimpleNamespace(**namespace),
            'NEOPAX._geometry_models': SimpleNamespace(VmecBoozer=SimpleNamespace),
            'NEOPAX._state': SimpleNamespace(TransportState=SimpleNamespace),
            'NEOPAX._transport_flux_models': SimpleNamespace(NTXRuntimeScanTransportModel=Neo),
        }
        database_source = ast.parse((SCRIPT.parents[2]/'NEOPAX/_database.py').read_text())
        database_class = next(node for node in database_source.body
                              if isinstance(node, ast.ClassDef) and node.name == 'Monoenergetic')
        saved = {node.target.id:np.asarray([i+0.25]) for i, node in enumerate(database_class.body)
                 if isinstance(node, ast.AnnAssign)}
        ad = {"support['database']."+k:v for k, v in saved.items()}
        ad["support['geometry'].a_b"] = np.asarray(1.2)
        fd = {f'initial_state.{k}':np.asarray([i+1.])
              for i, k in enumerate(('density', 'pressure', 'Er'))}
        build = Mock(return_value=Flux(Neo()))
        species = SimpleNamespace(names=('e','D','T','He'))
        orch = SimpleNamespace(_build_species=lambda cfg:species, _build_energy_grid=lambda cfg:object(),
            build_source_models_from_config=lambda *args:object(), _build_flux_model=build,
            RuntimeContext=SimpleNamespace, Models=SimpleNamespace, _normalize_solver_config=lambda cfg:{})
        with patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()):
            for mode, expected in ((None, Monoenergetic), ('generic', Monoenergetic),
                                  (' preprocessed_3d_radial_ntss1d ', MonoenergeticRuntimeNTSS1D)):
                with self.subTest(mode=mode):
                    config = {'neoclassical':{'flux_model':'ntx_scan_runtime', 'interpolation_mode':mode}}
                    runtime, state = check.restore_runtime(config, ad, fd, orch, np)
                    self.assertIs(type(runtime.database), expected)
                    self.assertIs(runtime.models.flux.neoclassical_model.database, runtime.database)
                    self.assertEqual(set(vars(runtime.database)), set(saved))
                    for key, value in saved.items():
                        np.testing.assert_array_equal(getattr(runtime.database, key), value)
                    np.testing.assert_array_equal(state.Er, fd['initial_state.Er'])
                    self.assertFalse(config['neoclassical']['prebuild_database'])
                    self.assertFalse(config['neoclassical']['preload_channels'])
            for settings in ({'flux_model':'ntx_database'},
                             {'flux_model':'ntx_scan_runtime', 'interpolation_mode':'unsupported'}):
                build.reset_mock()
                with self.assertRaises(ValueError):
                    check.restore_runtime({'neoclassical':settings}, ad, fd, orch, np)
                build.assert_not_called()

            # Aborted paired runs provide packed initial_y. Preserve the raw
            # zero-He condition instead of projecting from the final template.
            ad.update({'final_state.Er':np.asarray([99.,99.]),
                       'final_state.pressure':np.full((4,2),99.)})
            ions = np.asarray([[1.,2.],[3.,4.],[0.,1e-12]])
            pressure = np.arange(8,dtype=float).reshape(4,2)
            packed = np.r_[ions.reshape(-1),pressure.reshape(-1),[-3.,-4.]]
            def neutralize(state,species):
                state.density[0] = state.density[1]+state.density[2]+2*state.density[3]
                return state
            with patch.dict(sys.modules, {'NEOPAX._transport_equations':SimpleNamespace(
                    enforce_quasi_neutrality=neutralize)}):
                _, state = check.restore_runtime({'neoclassical':{'flux_model':'ntx_scan_runtime'}},
                    ad, {'initial_y':packed}, orch, np)
            np.testing.assert_array_equal(state.density[1:],ions)
            np.testing.assert_array_equal(state.density[0],ions[0]+ions[1]+2*ions[2])
            np.testing.assert_array_equal(state.pressure,pressure)
            np.testing.assert_array_equal(state.Er,[-3.,-4.])

    def test_fd_endpoint_profile_wrapper_keeps_its_existing_selected_root_rule(self):
        path = SCRIPT.with_name('benchmark_transport_realtime_geometry_forward_fd.py')
        function = next(node for node in ast.parse(path.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == '_profile_state_from_values')
        constructor = Mock(return_value=object())
        namespace = dict(_initial_state_for_parameter_vector=constructor)
        future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[])), str(path), 'exec'), namespace)
        values, runtime, baseline = object(), object(), object()
        kwargs = dict(config={}, runtime=runtime, baseline_state=baseline, profile_cfg={},
                      initial_er_root_ad='jax_selected_root')
        result = namespace['_profile_state_from_values'](values, **kwargs)
        self.assertIs(result, constructor.return_value)
        self.assertIs(constructor.call_args.kwargs['runtime'], runtime)
        self.assertNotIn('known_initial_er_profile', constructor.call_args.kwargs)
        self.assertEqual(constructor.call_args.kwargs['initial_er_root_ad'], 'jax_selected_root')

    def test_fd_baseline_uses_actual_ad_schedule_constructor_not_outer_runtime_cache(self):
        path = SCRIPT.with_name('benchmark_transport_realtime_geometry_forward_fd.py')
        module = ast.parse(path.read_text())
        main = next(node for node in module.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'main')
        start = next(i for i, node in enumerate(main.body) if isinstance(node, ast.Assign)
                     and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'baseline_profile_builder')
        end = next(i for i, node in enumerate(main.body[start:], start) if isinstance(node, ast.Assign)
                   and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'baseline_profile_state')
        imports = next(node for node in module.body if isinstance(node, ast.ImportFrom)
                       and node.module == 'NEOPAX._reverse_ad_transport')
        self.assertIn('initial_state_for_parameter_vector', [name.name for name in imports.names])
        @dataclasses.dataclass
        class State:
            Er: object
        preliminary = State(np.asarray([1., -2.]))
        root = np.asarray([1.001, -2.001])
        constructor = Mock(return_value=State(root))
        builder = Mock(return_value=preliminary)
        namespace = dict(_profile_state_from_values=builder, profile_values=object(),
            config={'neoclassical':{'flux_model':'ntx_scan_runtime'}}, root_only_fd=False,
            initial_er_root_ad='jax_selected_root', baseline_runtime=object(),
            baseline_state=preliminary, profile_cfg={}, initial_state_for_parameter_vector=constructor)
        code = compile(ast.Module(body=main.body[start:end+1], type_ignores=[]), str(path), 'exec')
        exec(code, namespace)
        builder.assert_not_called()
        self.assertEqual(constructor.call_args.kwargs['initial_er_root_ad'], 'jax_selected_root')
        self.assertNotIn('known_initial_er_profile', constructor.call_args.kwargs)
        fd_state = namespace['baseline_profile_state']
        fd_call = constructor.call_args

        # Execute the constructor call that records the AD adaptive schedule.
        transport = path.parents[2]/'NEOPAX/_reverse_ad_transport.py'
        evaluator = next(node for node in ast.parse(transport.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == 'prepare_reverse_static_setup')
        stage = next(node for node in evaluator.body if isinstance(node, ast.Assign)
                     and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'state0_static')
        ad = dict(initial_state_for_parameter_vector=constructor, parameter_values=namespace['profile_values'],
            config=namespace['config'], runtime=namespace['baseline_runtime'], baseline_state=preliminary,
            profile_cfg=namespace['profile_cfg'], initial_er_root_ad='jax_selected_root')
        exec(compile(ast.Module(body=[stage], type_ignores=[]), str(transport), 'exec'), ad)
        self.assertEqual(fd_call, constructor.call_args)
        np.testing.assert_array_equal(fd_state.Er, ad['state0_static'].Er)
        self.assertFalse(np.array_equal(fd_state.Er, preliminary.Er))

        for flux, root_only, mode in (
            ('ntx_database', False, 'jax_selected_root'),
            ('ntx_scan_runtime', True, 'jax_selected_root'),
            ('ntx_scan_runtime', False, 'off'),
        ):
            namespace['config']['neoclassical']['flux_model'] = flux
            namespace['root_only_fd'] = root_only
            namespace['initial_er_root_ad'] = mode
            constructor.reset_mock()
            exec(code, namespace)
            constructor.assert_not_called()
            self.assertEqual(builder.call_args.kwargs['initial_er_root_ad'], 'off' if root_only else mode)

    def test_actual_evaluator_root_setup_uses_hybrid_without_changing_runtime_toml(self):
        path = SCRIPT.parents[2]/'NEOPAX/_reverse_ad_initial_er.py'
        function = next(node for node in ast.parse(path.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == 'initial_er_root_setup')
        namespace = dict(get_entropy_model=Mock(return_value=object()))
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        config = {'ambipolarity':{'er_ambipolar_scan_batch_mode':'vmap', 'er_ambipolar_blocksize':1}}
        runtime = SimpleNamespace(species=object(), energy_grid=object(), geometry=object(),
                                  database=object(), solver_parameters={})
        amb, method, entropy, params = namespace['initial_er_root_setup'](config, runtime)
        self.assertEqual(amb['er_ambipolar_scan_batch_mode'], 'hybrid')
        self.assertEqual(config['ambipolarity']['er_ambipolar_scan_batch_mode'], 'vmap')
        self.assertEqual(method, 'two_stage')
        self.assertIs(params['geometry'], runtime.geometry)

    def test_exact_comparison_catches_ulp_signed_zero_and_first_attempt(self):
        a = np.asarray([0., 1., 2., 3., 4., 5.])
        b = a.copy()
        b[5] = np.nextafter(b[5], np.inf)
        rows = check.differences({"err_norms": a}, {"err_norms": b})
        self.assertEqual(rows[0]["first_attempt"], 6)
        self.assertEqual(rows[0]["different_entries"], 1)
        self.assertEqual(rows[0]["max_abs"], b[5] - a[5])
        self.assertEqual(check.differences({"a": a}, {"a": a.copy()}), [])
        b = a.copy()
        b[0] = -0.
        row = check.differences({"a": a}, {"a": b})[0]
        self.assertEqual(row["max_abs"], 0.)
        self.assertEqual(row["first_flat_index"], 0)
        self.assertEqual(_jax_before, _jax_after)

    def test_nonfinite_dtype_shape_and_missing_fields_are_visible(self):
        a, b = np.asarray([np.inf, np.nan, 2.]), np.asarray([np.inf, np.nan, 3.])
        row = check.differences({"a": a}, {"a": b})[0]
        self.assertTrue(row["finite_masks_match"])
        self.assertEqual(row["first_flat_index"], 2)
        self.assertEqual(row["max_abs"], 1.)
        b[0] = 0.
        self.assertFalse(check.differences({"a": a}, {"a": b})[0]["finite_masks_match"])
        for other in ({}, {"a": a.astype(np.float32)}, {"a": a[:2]}):
            self.assertIn("reason", check.differences({"a": a}, other)[0])

    def test_leaf_capture_and_file_selection_preserve_data_without_calling_functions(self):
        @dataclasses.dataclass
        class Carry:
            y: object
            previous: object
            rhs: object
        def forbidden():
            self.fail("Capturing setup must not evaluate the RHS.")
        carry = Carry(np.asarray([1., 2.]), (np.asarray(3.), None), forbidden)
        leaves = check.numeric_leaves(carry, "carry")
        self.assertEqual(set(leaves), {"carry.y", "carry.previous[0]"})
        carry.y[0] = 9.
        self.assertEqual(leaves["carry.y"][0], 1.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaises(ValueError):
                check.read_one(path, "*.npz")
            np.savez(path / "one.npz", **leaves)
            self.assertEqual(check.differences(leaves, check.read_one(path, "*.npz")), [])
            np.savez(path / "two.npz", y=np.ones(1))
            with self.assertRaises(ValueError):
                check.read_one(path, "*.npz")
            check.write_json(path / "metadata.json", {"value": np.float64(2.)})
            self.assertEqual(json.loads((path / "metadata.json").read_text()), {"value": 2.})

    def test_fresh_capture_saves_existing_setup_and_active_schedule_only(self):
        @dataclasses.dataclass
        class Carry:
            y: object
            t: object
        @dataclasses.dataclass
        class Runtime:
            geometry: object
            solver_parameters: dict
        weak = np.asarray([1., 2.]).view(type("WeakArray", (np.ndarray,), {"weak_type": True}))
        prepared = SimpleNamespace(initial_carry=Carry(weak, np.asarray(0.)))
        components = {"equation_system": SimpleNamespace(fixed_temperature_profile=np.ones((2, 3)))}
        trace = SimpleNamespace(**{name: np.arange(8.) for name in check.SCHEDULE_FIELDS})
        trace.jacobian_reused = None
        rollout = SimpleNamespace(trace=trace, final_carry=Carry(np.asarray([3., 4.]), np.asarray(0.2)),
                                  attempt_count=3, accepted_count=2)
        runtime = Runtime(np.asarray([8.]), {"rtol": 1e-6})
        with tempfile.TemporaryDirectory() as directory, patch.object(check, "version", return_value="test"):
            check.capture_fresh_baseline(directory, components=components, prepared=prepared,
                execution={"dt_min": np.asarray(1e-14)}, rollout=rollout,
                runtime=runtime, state=prepared.initial_carry, parameter_name="ZBS:1:0")
            directory = Path(directory)
            setup = check.read_one(directory, "setup.npz")
            self.assertTrue(setup["initial_carry.y.__weak_type__"])
            np.testing.assert_array_equal(setup["initial_carry.y"], [1., 2.])
            np.testing.assert_array_equal(setup["fixed_temperature_profile"], np.ones((2, 3)))
            native = check.read_one(directory, "native.npz")
            np.testing.assert_array_equal(native["err_norms"], [0., 1., 2.])
            np.testing.assert_array_equal(native["final_y"], [3., 4.])
            self.assertNotIn("jacobian_reused", native)
            metadata = json.loads((directory / "metadata.json").read_text())
            self.assertEqual(metadata["accepted_count"], 2)
            self.assertEqual(metadata["parameter_name"], "ZBS:1:0")

    def test_original_fd_baseline_only_stops_after_unchanged_schedule_call(self):
        source = SCRIPT.with_name("benchmark_transport_realtime_geometry_forward_fd.py")
        main = next(node for node in ast.parse(source.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "main")
        runtime, state, prepared, execution = (object() for _ in range(4))
        rollout = SimpleNamespace(trace=object())
        components = {"solver": object()}
        config = {"geometry": {}}
        schedule = Mock(return_value=(components, prepared, execution, rollout, object()))
        capture = Mock()
        objectives = Mock(side_effect=AssertionError("ordinary objective path reached"))
        namespace = dict(argparse=argparse, Path=Path, contextlib=contextlib,
            _execution_device_context=lambda config: contextlib.nullcontext(),
            atexit=SimpleNamespace(register=Mock()), jnp=np,
            jax=SimpleNamespace(default_backend=lambda: "gpu", local_devices=lambda **kw: ["cuda:0"],
                                default_device=lambda device: contextlib.nullcontext()),
            DEFAULT_CONFIG=Path("unused.toml"), PARAMETER_ORDER=("n0",),
            INITIAL_ER_ROOT_ONLY_EXPLICIT_OBJECTIVES=(),
            _prepare_benchmark_config=Mock(return_value=config),
            _initial_er_root_ad_mode=lambda mode: mode,
            _baseline_profile_cfg=lambda cfg: {}, _profile_cfg_scalar_value=lambda *a: 2.,
            _frozen_linearized_vmec_geometry_bundle=Mock(return_value={}),
            _runtime_for_frozen_linearized_geometry_step=Mock(return_value=(runtime, state)),
            _profile_state_from_values=Mock(return_value=state), _schedule_rollout=schedule,
            _truncate_rollout_trace_by_accepted_steps=lambda *a: None,
            _objective_vector=objectives)
        exec(compile(ast.Module(body=[main], type_ignores=[]), str(source), "exec"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            arguments = ["fd", "--parameter", "ZBS:1:0", "--accepted-step-limit", "16",
                         "--fd-accepted-replay-contract", "reverse_frozen_metadata"]
            with patch.dict(sys.modules, {"check_saved_transport_baseline_repeatability":
                                          SimpleNamespace(capture_fresh_baseline=capture)}), \
                 patch.object(sys, "argv", arguments + ["--ad-geometry-reference", directory,
                                                        "--baseline-only-dir", directory]), \
                 contextlib.redirect_stdout(io.StringIO()):
                namespace["main"]()
            schedule.assert_called_once_with(config, runtime, state,
                                            accepted_step_limit=16, capture_newton_setup=True)
            namespace['_frozen_linearized_vmec_geometry_bundle'].assert_called_once_with(
                config,'ZBS:1:0',baseline_delta=0.,ad_geometry_reference=Path(directory))
            self.assertEqual(capture.call_args.kwargs["parameter_name"], "ZBS:1:0")
            self.assertIs(capture.call_args.kwargs["prepared"], prepared)
            self.assertIs(capture.call_args.kwargs["rollout"], rollout)
            objectives.assert_not_called()
            # With no diagnostic flag, normal execution still reaches the objective path.
            capture.reset_mock()
            with patch.object(sys, "argv", arguments), contextlib.redirect_stdout(io.StringIO()), \
                 self.assertRaisesRegex(AssertionError, "ordinary objective path reached"):
                namespace["main"]()
            capture.assert_not_called()

            # Exercise the actual CLI path with an AD reference and prove that
            # a changed terminal carry stops before objective/FD evaluation.
            @dataclasses.dataclass
            class Geo:
                scale: object
            @dataclasses.dataclass
            class Database:
                table: object
            @dataclasses.dataclass
            class State:
                density: object
                pressure: object
                Er: object
            state = State(np.asarray([[1., 2.]]), np.asarray([[3., 4.]]),
                          np.asarray([5., np.nextafter(6., np.inf)]))
            runtime = SimpleNamespace(geometry=Geo(np.asarray(1.2)), species=object(),
                models=SimpleNamespace(flux=SimpleNamespace(neoclassical_model=
                    SimpleNamespace(database=Database(np.asarray([1., 2.]))))))
            ref = {'initial_y':np.arange(1., 7.), 'final_y':np.arange(2., 8.),
                   'final_time':np.asarray(1e-4), "support['geometry'].scale":np.asarray(1.2),
                   "support['database'].table":np.asarray([1., 2.])}
            ref_path = Path(directory)/'ad_reference.npz'
            np.savez(ref_path, **ref)
            def schedule_reference(cfg, rt, selected, **kwargs):
                np.testing.assert_array_equal(selected.Er, [5., 6.])
                np.testing.assert_array_equal(selected.density, state.density)
                return (components, SimpleNamespace(initial_carry=SimpleNamespace(y=ref['initial_y'])),
                        execution, SimpleNamespace(trace=object(), final_carry=SimpleNamespace(
                            y=ref['final_y'].copy(), t=ref['final_time'])), object())
            namespace.update(np=np, dataclasses=dataclasses, save_gradient_audit=Mock(),
                _pack_transport_state_arrays=lambda s, sp:(s.density, s.pressure, s.Er),
                _runtime_for_frozen_linearized_geometry_step=Mock(return_value=(runtime, state)),
                _profile_state_from_values=Mock(return_value=state),
                _schedule_rollout=schedule_reference)
            module = SimpleNamespace(capture_fresh_baseline=capture,
                ad_reference_initial_er=check.ad_reference_initial_er,
                require_ad_reference_baseline=check.require_ad_reference_baseline)
            ref_args = arguments + ['--initial-Er-root-ad', 'jax_selected_root',
                '--initial-Er-root-fd-root-lane', 'frozen_linearized',
                '--ad-baseline-reference', str(ref_path), '--baseline-only-dir', directory]
            objectives.reset_mock()
            with patch.dict(sys.modules, {'check_saved_transport_baseline_repeatability':module}), \
                 patch.object(sys, 'argv', ref_args), contextlib.redirect_stdout(io.StringIO()):
                namespace['main']()
                capture.reset_mock()
                original_schedule = namespace['_schedule_rollout']
                def mismatched_schedule(*args, **kwargs):
                    parts = original_schedule(*args, **kwargs)
                    parts[3].final_carry.y[0] = np.nextafter(parts[3].final_carry.y[0], np.inf)
                    return parts
                namespace['_schedule_rollout'] = mismatched_schedule
                with self.assertRaisesRegex(ValueError, 'before FD endpoints'):
                    namespace['main']()
                capture.assert_not_called()
                objectives.assert_not_called()

            # A support mismatch must be saved and rejected before time stepping.
            runtime.models.flux.neoclassical_model.database.table[0] += .125
            audit = namespace['save_gradient_audit']
            audit.reset_mock()
            schedule = Mock(side_effect=AssertionError('time evolution reached'))
            namespace['_schedule_rollout'] = schedule
            with patch.dict(sys.modules, {'check_saved_transport_baseline_repeatability':module}), \
                 patch.object(sys, 'argv', ref_args), contextlib.redirect_stdout(io.StringIO()), \
                 self.assertRaisesRegex(ValueError, 'geometry/database'):
                namespace['main']()
            schedule.assert_not_called()
            audit.assert_called_once()
            self.assertEqual(audit.call_args.args,('fd_reference_support',))
            self.assertIs(audit.call_args.kwargs['support']['database'],runtime.models.flux.neoclassical_model.database)


if __name__ == "__main__":
    unittest.main()
