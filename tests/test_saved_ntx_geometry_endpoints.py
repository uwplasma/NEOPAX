"""Independent endpoint error budgets and pre-NTX stop; no JAX import."""
import ast
from contextlib import redirect_stdout
import csv
import dataclasses
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1]/'examples/benchmarks/check_saved_ntx_geometry_endpoints.py'
sys.path.insert(0,str(SCRIPT.parent))
loaded_jax = 'jax' in sys.modules
spec = importlib.util.spec_from_file_location('endpoint_check',SCRIPT)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class EndpointTests(unittest.TestCase):
    def test_budget_separates_upstream_nonlinearity_and_context_from_saved_drift(self):
        # Independent composition: table=input^3, input=x^3. A finite input
        # secant differs from its tangent, separately from coefficient FD bias.
        x,h = .7,.03
        m,p = (x-h)**3,(x+h)**3
        reference = 9*x**8
        upstream = 3*(x**3)**2*(p-m)/(2*h)
        scalar_minus,scalar_plus = m**3,p**3
        # Inject known errors into two later boundaries, independently of the
        # nonlinear FD bias. No single error may absorb the other three.
        scan_minus,scan_plus = scalar_minus-.2*h,scalar_plus+.2*h
        saved_minus,saved_plus = scan_minus-.4*h,scan_plus+.4*h
        vector = lambda v: np.asarray([v,2*v,-v])
        values = dict(saved_minus=vector(saved_minus),saved_plus=vector(saved_plus),
            scan_minus=vector(scan_minus),scan_plus=vector(scan_plus),
            scalar_minus=vector(scalar_minus),scalar_plus=vector(scalar_plus),
            input_secant_jvp=vector(upstream),control_jvp=vector(reference))
        rows = check.gap_rows((3,4,5),vector(reference),values,h)
        for i,scale in enumerate((1.,2.,-1.)):
            r = rows[i]
            self.assertAlmostEqual(r['upstream_gap'],scale*(reference-upstream))
            self.assertNotEqual(r['upstream_gap'],0.)
            self.assertNotEqual(r['scalar_fd_gap'],0.)
            self.assertAlmostEqual(r['scan_context_gap'],-scale*.2)
            self.assertAlmostEqual(r['reproduction_gap'],-scale*.4)
            self.assertEqual(r['audit_control_gap'],0.)
            self.assertAlmostEqual(sum(r[k] for k in ('upstream_gap','scalar_fd_gap','scan_context_gap','reproduction_gap')),r['total_gap'])
        for invalid in (0.,float('nan')):
            with self.assertRaisesRegex(ValueError,'step'):
                check.gap_rows((0,0,0),vector(reference),values,invalid)
        with self.assertRaisesRegex(ValueError,'finite'):
            check.gap_rows((0,0,0),vector(reference),dict(values,input_secant_jvp=np.asarray([np.nan]*3)),h)

    def test_original_entrypoint_is_used_and_stopped_before_any_ntx_or_root_solve(self):
        calls = []
        original = lambda context,state,geometry,**kwargs: (calls.append(('inputs',state)) or ('channels','surfaces'))
        geo = SimpleNamespace(build_ntx_runtime_scan_inputs_from_vmec_state=original)
        def runtime(config,bundle,*,step_scale):
            calls.append(('entrypoint',step_scale))
            geo.build_ntx_runtime_scan_inputs_from_vmec_state('context','perturbed state','geometry',rho_scan=[.5])
            self.fail('NTX and root construction must never be reached')
        forward = SimpleNamespace(_runtime_for_frozen_linearized_geometry_step=runtime)
        result = check.collect_inputs(forward,geo,{}, {},-.1)
        self.assertEqual(result,dict(geometry='geometry',channels='channels',surfaces='surfaces'))
        self.assertEqual(calls,[('entrypoint',-.1),('inputs','perturbed state')])
        self.assertIs(geo.build_ntx_runtime_scan_inputs_from_vmec_state,original)
        bundle = {}
        baseline = check.collect_inputs(forward,geo,{},bundle,0.)
        self.assertIs(bundle['native_baseline_surfaces'],baseline['surfaces'])
        with self.assertRaisesRegex(RuntimeError,'pre-NTX'):
            check.collect_inputs(SimpleNamespace(_runtime_for_frozen_linearized_geometry_step=lambda *a,**kw:None),geo,{}, {},0.)

    def test_cli_budget_uses_full_scan_shape_actual_endpoints_and_live_normalization(self):
        State = dataclasses.make_dataclass('State',[('x',object)])
        Geometry = dataclasses.make_dataclass('Geometry',[('n_r',int),('x',object)])
        Surface = dataclasses.make_dataclass('Surface',[(n,object) for n in ('m','n','nfp','b_cos','iota')])
        x,h,dx = .7,1e-4,.3
        def inputs(v):
            s = Surface(np.asarray([0]),np.asarray([0]),1,np.asarray(v**3),np.asarray(.1*v))
            c = dict(drds=np.asarray([2+v*v]),b00=np.asarray([2+v]),dr_tildeds=np.asarray([1+v*v]))
            return s,c
        field_channels = lambda rho,er,c:(None,3*c['dr_tildeds'][:,None]*c['b00'][:,None],None)
        def coeff(s,nu,es):
            return [s.b_cos**2+s.iota,0.,s.b_cos*es+s.iota**2,es,0.]
        def normalized(v,context=False):
            s,c = inputs(v)
            es = field_channels(None,None,c)[1][0,0]
            d11,_,d13,d33,_ = coeff(s,.7,es)
            if context:
                d13 += es*1e-5
            dr = c['drds'][0]
            return np.asarray([np.log10(d11*dr*dr),d13*dr,d33*.7])
        reference = normalized(x+1e-25j).imag/1e-25*dx
        def scan_builder(context,state,geometry,**kwargs):
            s,c = inputs(state.x[0])
            return c,(s,)
        geo = SimpleNamespace(build_ntx_runtime_scan_inputs_from_vmec_state=scan_builder)
        geo._traceable_vmec_surfaces_from_state = lambda context,state,**kw:(inputs(state.x[0])[0],)
        geo._positive_transport_s_values_from_rho = lambda rho:rho
        steps,shapes = [],[]
        def runtime(config,bundle,*,step_scale):
            steps.append(step_scale)
            state = State(bundle['state_star'].x+step_scale*bundle['state_tangent'].x)
            geo.build_ntx_runtime_scan_inputs_from_vmec_state(None,state,Geometry(1,state.x[0]))
            self.fail('Root or transport work reached')
        def scan(s,grid,nu,*,epsi_hat):
            shapes.append(nu.shape)
            c = coeff(s,nu,epsi_hat)
            return dict(D11=c[0]+np.zeros_like(nu),D13=c[2]+epsi_hat*1e-5,D33=c[3])
        def jvp(f,p,t):
            if isinstance(p[0],State):
                dot = ((3*p[0].x[0]**2*t[0].x[0],.1*t[0].x[0]),)
            else:
                dot = np.imag(f(tuple(v+1e-25j*d for v,d in zip(p[0],t[0]))))/1e-25
            return f(*p),dot
        fake_jax = SimpleNamespace(numpy=np,default_backend=lambda:'gpu',config=SimpleNamespace(x64_enabled=True),
            devices=lambda:['test'],__version__='mock',jit=lambda f:f,block_until_ready=lambda v:v,jvp=jvp)
        modules = {'jax':fake_jax,'jax.numpy':np,
            'ntx':SimpleNamespace(GridSpec=lambda **kw:kw,prepare_monoenergetic_system=lambda s,g:s,
                MonoenergeticCase=lambda **kw:SimpleNamespace(**kw),solve_monoenergetic_scan=scan,
                solve_prepared_coefficient_vector=lambda s,c:np.asarray(coeff(s,c.nu_hat,c.epsi_hat))),
            'NEOPAX':SimpleNamespace(_geometry_autodiff=geo),
            'NEOPAX._transport_flux_models':SimpleNamespace(D11_POSITIVE_FLOOR=1e-20,_build_ntx_field_channels=field_channels),
            'vmex.core.solver':SimpleNamespace(SpectralState=State),
            'benchmark_transport_realtime_geometry_forward_fd':SimpleNamespace(_geometry_context_from_config=lambda *a:None,
                _runtime_for_frozen_linearized_geometry_step=runtime)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ad,fd = root/'ad/snapshots',root/'fd_ZBS_1_0'
            ad.mkdir(parents=True); (fd/'snapshots').mkdir(parents=True)
            config = dict(geometry=dict(vmec_input_file=str(root/'input'),n_radial=1),
                neoclassical=dict(ntx_scan_rho=[.5],ntx_scan_er_tilde=[3.],ntx_scan_nu_v=[.7],
                    ntx_scan_n_theta=1,ntx_scan_n_zeta=1,ntx_scan_n_xi=1))
            (root/'config.toml').write_text(''); (root/'input').write_text('input')
            modules['benchmark_transport_autodiff_lagged_ntx'] = SimpleNamespace(_prepare_benchmark_config=lambda *a,**kw:config)
            base = {k:np.asarray([1.]) for k in ('initial_y','final_y','final_time','values')}
            base.update({"support['geometry'].n_r":np.asarray(1),"support['geometry'].x":np.asarray(x),
                         'labels[0]':np.asarray('cost')})
            base.update({"support['database']."+k:np.asarray([[[v]]]) for k,v in zip(check.TABLE_FIELDS,normalized(x,True))})
            np.savez(ad/'ad_terminal_one.npz',**base)
            np.savez(fd/'snapshots/fd_baseline_one.npz',**base)
            for sign,mult in (('minus',-1),('plus',1)):
                np.savez(fd/f'snapshots/fd_ZBS_{sign}_one.npz',**{"support['database']."+k:np.asarray([[[v]]]) for k,v in zip(check.TABLE_FIELDS,normalized(x+mult*h*dx,True))})
            np.savez(ad/'ad_compact_geometry_one.npz',**{'state.x':np.asarray([x])})
            payload = {"['surfaces'][0].b_cos":np.asarray(3*x*x*dx),
                "['channels'].drds":np.asarray([2*x*dx]),"['channels'].dr_tildeds":np.asarray([2*x*dx]),
                "['channels'].b00":np.asarray([dx])}
            tangent_values = {'parameter_labels[0]':np.asarray('ZBS:1:0'),'state_tangents.x':np.asarray([[dx]])}
            for i,(k,v) in enumerate(payload.items()):
                tangent_values[f'payload_paths[{i}]'] = np.asarray(k)
                tangent_values[f'payload_tangents[{i}]'] = np.asarray([v])
            np.savez(ad/'ad_compact_tangents_one.npz',**tangent_values)
            np.savez(ad/'ad_before_scan_fold_one.npz',**{'labels[0]':np.asarray('cost'),**{f"support_bars[0]['database'].{k}":np.ones((1,1,1)) for k in check.TABLE_FIELDS}})
            np.savez(root/'directions.npz',parameter=np.asarray('ZBS:1:0'),**{'ad.x':np.asarray([dx]),'fd.x':np.asarray([dx])})
            (fd/'summary.json').write_text(json.dumps(dict(parameter_name='ZBS:1:0',fd_step=h,
                geometry_fd_lane='frozen_linearized',objective_labels=['cost'])))
            ref = {'database_dot.'+k:np.asarray([[[v]]]) for k,v in zip(check.TABLE_FIELDS,reference)}
            argv = [str(SCRIPT),'--run-dir',str(root),'--out-dir',str(root/'result'),
                '--state-tangents',str(root/'directions.npz'),'--table-audit',str(root/'table'),'--points','0:0:0']
            with patch.dict(sys.modules,modules),patch.object(sys,'argv',argv), \
                 patch.object(check,'precision_reference',return_value=({},ref)), \
                 patch.object(check.importlib,'import_module',return_value=SimpleNamespace(__file__=__file__)),redirect_stdout(io.StringIO()):
                check.main()
            with (root/'result/endpoint_gaps.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(steps,[0.,-h,h])
            self.assertEqual(shapes,[(1,1)]*3)
            self.assertEqual(len(rows),3)
            self.assertLess(max(abs(float(r['reproduction_gap'])) for r in rows),1e-9)
            d13 = rows[1]
            self.assertGreater(abs(float(d13['scan_context_gap'])),1e-5)
            self.assertLess(abs(float(d13['upstream_gap'])),1e-7)
            self.assertLess(max(abs(float(r['audit_control_gap'])) for r in rows),1e-12)
            self.assertTrue((root/'result/selected_node_budget.csv').is_file())
            with np.load(root/'result/minus_inputs.npz') as z:
                np.testing.assert_array_equal(z['surfaces[0].b_cos'],(x-h*dx)**3)

    def test_exact_baseline_guard_rejects_signed_zero_dtype_shape_and_missing_data(self):
        a = np.asarray([0.,2.])
        check.require_equal({'x':a.copy()},{'x':a},'baseline')
        for wrong in (np.asarray([-0.,2.]),a.astype(np.float32),a[:,None],np.asarray([0.,np.nan])):
            with self.assertRaisesRegex(ValueError,'differs'):
                check.require_equal({'x':wrong},{'x':a},'baseline')
        with self.assertRaisesRegex(ValueError,'Missing'):
            check.require_equal({}, {'x':a},'baseline')
        self.assertEqual('jax' in sys.modules,loaded_jax)
        tree = ast.parse(SCRIPT.read_text())
        imports = [n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom))]
        self.assertFalse(any('jax' in ast.unparse(n) for n in imports))
        calls = {n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
        self.assertIn('solve_monoenergetic_scan',calls)
        self.assertNotIn('geometry_raw_block_solve_from_param_vector',calls)


if __name__ == '__main__':
    unittest.main()
