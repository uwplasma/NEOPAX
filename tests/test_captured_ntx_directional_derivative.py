"""Saved-array preparation and API contracts; import no physics packages."""
import ast
import csv
from contextlib import redirect_stdout
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

SCRIPT = Path(__file__).resolve().parents[1]/'examples/benchmarks/check_captured_ntx_directional_derivative.py'
JAX_ALREADY_LOADED = 'jax' in sys.modules
spec = importlib.util.spec_from_file_location('captured_ntx_check', SCRIPT)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)
JAX_AFTER_IMPORT = 'jax' in sys.modules


class CapturedNTXPreparationTests(unittest.TestCase):
    def test_captured_objective_bars_include_raw_normalization_and_d11_floor(self):
        before = {f'labels[{i}]':np.asarray(name) for i,name in enumerate(('Er','bootstrap'))}
        weights = ((2.,3.,-4.),(-1.,5.,6.))
        for i,row in enumerate(weights):
            before.update({f"support_bars[{i}]['database'].{name}":np.asarray([[[value]]])
                           for name,value in zip(('D11_log','D13','D33'),row,strict=True)})
        raw = np.asarray([1.2,0.,3.4,5.6,0.])
        dot = np.asarray([.4,0.,.6,.8,0.])
        bars,labels = check.captured_coefficient_bars(before,(0,0,0),raw,2.,.3)
        self.assertEqual(labels,['Er','bootstrap'])
        np.testing.assert_allclose(bars@dot,[a*.4/(np.log(10.)*1.2)+b*.6*2.+c*.8*.3
                                           for a,b,c in weights],rtol=1e-15)
        np.testing.assert_array_equal(bars[:,(1,4)],0.)
        raw[0] = -1.
        floored,_ = check.captured_coefficient_bars(before,(0,0,0),raw,2.,.3)
        np.testing.assert_array_equal(floored[:,0],0.)
        np.testing.assert_array_equal(floored[:,(2,3)],bars[:,(2,3)])
        with self.assertRaisesRegex(ValueError,'Invalid raw'):
            check.captured_coefficient_bars(before,(0,0,0),raw[:-1],2.,.3)
        self.assertEqual(JAX_ALREADY_LOADED,'jax' in sys.modules)

    def test_derivative_only_performs_no_fd_evaluations_and_default_retains_existing_scales(self):
        values = (np.asarray(2.),np.asarray(-.5))
        direction = (np.asarray(.3),np.asarray(.2))
        calls = []
        def evaluate(x):
            calls.append(x)
            return np.asarray([x[0]*x[1],x[0]**2])
        self.assertEqual(list(check.local_fd_samples(evaluate,values,direction,1e-5,jvp_vjp_only=True)),[(None,None)])
        self.assertEqual(calls,[])
        samples = list(check.local_fd_samples(evaluate,values,direction,1e-5))
        self.assertEqual(len(calls),6)
        self.assertEqual([h for h,_ in samples],[1e-5*scale for scale in (10.,3.,1.)])
        for step,((h,derivative),scale) in enumerate(zip(samples,(10.,3.,1.),strict=True)):
            for sign,actual in zip((-1,1),calls[2*step:2*step+2],strict=True):
                np.testing.assert_array_equal(actual,tuple(x+sign*scale*1e-5*d for x,d in zip(values,direction,strict=True)))
            np.testing.assert_allclose(derivative,[.25,1.2],rtol=1e-9,atol=1e-10)
        with self.assertRaisesRegex(FloatingPointError,'Nonfinite local'):
            list(check.local_fd_samples(lambda x:np.asarray([np.nan]),values,direction,1e-5))
        self.assertEqual(JAX_ALREADY_LOADED,'jax' in sys.modules)

    def test_precision_reference_rejects_different_direction_point_and_database(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            terminal = {'labels[0]':np.asarray('cost'),"support['database'].D13":np.asarray([2.])}
            np.savez(root/'raw_direction.npz',active_points=np.asarray([[0,0,0]]))
            np.savez(root/'database_direction.npz',**{'database.D13':np.asarray([2.])})
            (root/'table_comparison.csv').write_text('objective,parameter\ncost,ZBS:1:0\n')
            check.precision_reference(root,terminal,'ZBS:1:0',((0,0,0),))
            for parameter,points in (('RBC:1:0',((0,0,0),)),('ZBS:1:0',((1,0,0),))):
                with self.assertRaises(ValueError):
                    check.precision_reference(root,terminal,parameter,points)
            np.savez(root/'database_direction.npz',**{'database.D13':np.asarray([3.])})
            with self.assertRaisesRegex(ValueError,'database differs'):
                check.precision_reference(root,terminal,'ZBS:1:0',((0,0,0),))

    def test_precision_sweep_includes_extra_surface_direction_and_live_normalization(self):
        # Independent analytic model: D11=b^2+iota, D13=b*Es+iota^2,
        # D33=iota^2+Es. Nonzero iota_dot tests the extra surface fields.
        names = check.SURFACE_FIELDS+check.EXTRA_SURFACE_FIELDS
        Surface = dataclasses.make_dataclass('Surface',[(name,object) for name in (*names,'m','n','nfp')])
        values = dict.fromkeys(names,np.asarray(0.))
        values.update(b_cos=np.asarray(2.),iota=np.asarray(.5),m=np.asarray([0]),n=np.asarray([0]),nfp=1)
        surface = Surface(**values)
        State = dataclasses.make_dataclass('State',[('x',object)])
        state = State(np.asarray(0.))
        native_dots = (tuple(np.asarray(.3 if name=='iota' else .2 if name=='b_cos' else 0.) for name in names),)
        geometry = SimpleNamespace(_traceable_vmec_surfaces_from_state=lambda *a,**kw:(surface,),
            _positive_transport_s_values_from_rho=lambda x:x)
        field_channels = lambda *a:(np.zeros((1,1)),np.asarray([[3.]]),np.zeros((1,1)))
        fake_jax = SimpleNamespace(numpy=np,jit=lambda f:f,block_until_ready=lambda x:x,
            jvp=lambda f,p,t:(f(*p),native_dots))
        def coefficients(s,case):
            b,i,es = s.b_cos,s.iota,case.epsi_hat
            return np.asarray([b*b+i,0.,b*es+i*i,i*i+es,0.])
        fake_ntx = SimpleNamespace(prepare_monoenergetic_system=lambda s,g:s,
            MonoenergeticCase=lambda **kw:SimpleNamespace(**kw),solve_prepared_coefficient_vector=coefficients)
        modules = {'jax':fake_jax,'jax.numpy':np,'ntx':fake_ntx,
            'NEOPAX':SimpleNamespace(_geometry_autodiff=geometry),
            'NEOPAX._transport_flux_models':SimpleNamespace(D11_POSITIVE_FLOOR=1e-20,_build_ntx_field_channels=field_channels)}
        directions = {f"['surfaces'][0].{name}":np.asarray(.2 if name=='b_cos' else 0.) for name in check.SURFACE_FIELDS}
        directions.update({"['channels'].dr_tildeds":np.asarray([.1]),"['channels'].b00":np.asarray([.2]),
                           "['channels'].drds":np.asarray([.1])})
        raw = {'ordinary':np.asarray([4.5,6.25,3.25]).reshape(3,1,1,1),
               'raw_jvp':np.asarray([1.1,2.7,1.2]).reshape(3,1,1,1)}
        expected = [(1.1/4.5+.1)/np.log(10.),6.025,.84]
        reference = {'database_dot.'+field:np.asarray([[[dot]]]) for field,dot in zip(check.TABLE_FIELDS,expected)}
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules,modules), \
             patch.object(check,'precision_reference',return_value=(raw,reference)), redirect_stdout(io.StringIO()):
            args = SimpleNamespace(precision_from_table_audit=Path(directory),out_dir=Path(directory),
                parameter='ZBS:1:0',points=((0,0,0),))
            check.precision_sweep(args,terminal={},tangents={'parameter_labels[0]':np.asarray('ZBS:1:0'),
                'state_tangents.x':np.asarray([.1])},directions=directions,
                nc={'ntx_scan_rho':[.5],'ntx_scan_nu_v':[.7],'ntx_scan_er_tilde':[3.]},context=None,state=state,
                channels={'b00':np.asarray([1.]),'dr_tildeds':np.asarray([1.]),'drds':np.asarray([2.])},
                surfaces=(surface,),grid=None,h0=1e-6)
            with (Path(directory)/'precision.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows),12)
            self.assertEqual({float(r['fd_scale']) for r in rows},{1.,3.,10.,30.})
            self.assertLess(max(float(r['normalized_relative_difference']) for r in rows),1e-7)

    def test_snapshot_layouts_and_both_live_table_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, 'snapshot directory'):
                check.snapshot_directory(root)
            archived = root/'ad_snapshots'
            archived.mkdir()
            self.assertEqual(check.snapshot_directory(root), archived)
            archived.rmdir()
            current = root/'ad/snapshots'
            current.mkdir(parents=True)
            self.assertEqual(check.snapshot_directory(root), current)
            archived.mkdir()
            with self.assertRaisesRegex(ValueError, 'snapshot directory'):
                check.snapshot_directory(root)
        for mode in ('generic', 'preprocessed_3d_radial_ntss1d'):
            self.assertEqual(check.table_interpolation_mode({'interpolation_mode':mode}), mode)
        with self.assertRaises(ValueError):
            check.table_interpolation_mode({'interpolation_mode':'unrelated'})

    def test_saved_table_direction_matches_independent_polynomial_and_rejects_invalid_inputs(self):
        # D13(p)=p^2 and Er_list(p)=3p; weighted derivative at p=.7 is 2p*2-3.
        p,h = .7,1e-5
        def values(x):
            return {"support['database'].D13":np.asarray([x*x]),
                    "support['database'].Er_list":np.asarray(3*x)}
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder/'snapshots').mkdir()
            minus,plus = values(p-h),values(p+h)
            a = folder/'snapshots/fd_ZBS_minus_one.npz'
            z = folder/'snapshots/fd_ZBS_plus_one.npz'
            np.savez(a,**minus)
            np.savez(z,**plus)
            database = {key.split('.')[-1]:value for key,value in values(p).items()}
            direction = check.saved_database_direction(folder,database,h)
            before = {"support_bars[0]['database'].D13":np.asarray([2.]),
                      "support_bars[0]['database'].Er_list":np.asarray(-1.)}
            self.assertAlmostEqual(check.weighted_table_direction(before,'support_bars[0]',direction),4*p-3,places=9)
            for invalid_h in (0.,float('nan')):
                with self.assertRaises(ValueError):
                    check.saved_database_direction(folder,database,invalid_h)
            broken = dict(plus)
            broken.pop("support['database'].D13")
            np.savez(z,**broken)
            with self.assertRaisesRegex(ValueError,'Missing endpoint'):
                check.saved_database_direction(folder,database,h)
            plus["support['database'].D13"] = np.asarray([np.nan])
            np.savez(z,**plus)
            with self.assertRaisesRegex(ValueError,'Invalid FD'):
                check.saved_database_direction(folder,database,h)
            before["support_bars[0]['database'].D13"] = np.asarray([np.nan])
            with self.assertRaisesRegex(ValueError,'Invalid table'):
                check.weighted_table_direction(before,'support_bars[0]',direction)

    def test_fresh_fd_reference_preserves_archived_reference_and_checks_parameter(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            archived=root/'fd_ZBS_1_0'
            fresh=root/'fresh'
            archived.mkdir()
            fresh.mkdir()
            (archived/'summary.json').write_text(json.dumps(dict(parameter_name='ZBS:1:0',fd_step=1e-7)))
            (fresh/'summary.json').write_text(json.dumps(dict(parameter_name='ZBS:1:0',fd_step=2e-7)))
            args=SimpleNamespace(run_dir=root,parameter='ZBS:1:0',fd_reference_dir=None)
            self.assertEqual(check.fd_reference(args),(archived,dict(parameter_name='ZBS:1:0',fd_step=1e-7)))
            args.fd_reference_dir=fresh
            self.assertEqual(check.fd_reference(args),(fresh,dict(parameter_name='ZBS:1:0',fd_step=2e-7)))
            self.assertEqual(json.loads((archived/'summary.json').read_text())['fd_step'],1e-7)
            args.parameter='RBC:1:0'
            with self.assertRaisesRegex(ValueError,'different parameter'):
                check.fd_reference(args)

    def test_named_parameter_direction_and_field_boundaries(self):
        values = {'parameter_labels[0]': np.asarray('RBC:1:0'),
                  'parameter_labels[1]': np.asarray('ZBS:1:0'),
                  'payload_paths[0]': np.asarray("['channels'].b00"),
                  'payload_tangents[0]': np.asarray([[1.,2.],[3.,4.]])}
        np.testing.assert_array_equal(
            check.captured_direction(values, 'ZBS:1:0')["['channels'].b00"], [3.,4.])
        with self.assertRaises(ValueError):
            check.captured_direction(values, 'missing')
        self.assertEqual(check.fields({'state.R_cos': 2, 'state_extra.R_cos': 3,
                                      'state.nested.value': 4}, 'state.'), {'R_cos': 2})

    def test_point_validation_and_safe_module_import(self):
        self.assertEqual(check.point_indices('4:4:2,4:4:3'), ((4,4,2),(4,4,3)))
        for invalid in ('-1:0:0', '4:2', 'a:2:3', '4:4:2,4:4:2'):
            with self.assertRaises(check.argparse.ArgumentTypeError):
                check.point_indices(invalid)
        self.assertEqual(JAX_AFTER_IMPORT, JAX_ALREADY_LOADED)
        tree = ast.parse(SCRIPT.read_text())
        calls = {node.func.attr for node in ast.walk(tree)
                 if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)}
        self.assertIn('solve_prepared_coefficient_vector', calls)
        self.assertIn('pullback_prepared_database_coefficients_to_vmec_primitive_multi_rhs', calls)
        names = {node.func.id for node in ast.walk(tree)
                 if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)}
        self.assertIn('_prepared_implicit_vjp_primal', names)
        self.assertEqual(check.COEFFICIENTS, (('D11',0),('D13',2),('D33',3)))

    def test_active_nodes_and_projection_use_every_row_and_named_direction(self):
        before={}
        for i in range(2):
            for name in check.TABLE_FIELDS:
                before[f"support_bars[{i}]['database'].{name}"]=np.zeros((2,2,3))
        before["support_bars[0]['database'].D11_log"][0,1,2]=3.
        before["support_bars[1]['database'].D33"][1,0,1]=-2.
        self.assertEqual(check.active_table_points(before,2),((0,1,2),(1,0,1)))
        path="['geometry'].Bsqav"
        before['support_bars[0]'+path]=np.asarray([2.,-1.])
        self.assertEqual(check.project_saved_support(before,'support_bars[0]',
                         {path:np.asarray([3.,4.])}),2.)
        before["support_bars[0]['geometry'].missing"]=np.asarray(0.)
        self.assertEqual(check.project_saved_support(before,'support_bars[0]',
                         {path:np.asarray([3.,4.])}),2.)
        before["support_bars[0]['geometry'].missing"]=np.asarray(1.)
        with self.assertRaisesRegex(ValueError,'Missing captured direction'):
            check.project_saved_support(before,'support_bars[0]',{path:np.asarray([3.,4.])})
        before.pop("support_bars[0]['geometry'].missing")
        with self.assertRaisesRegex(ValueError,'shape mismatch'):
            check.project_saved_support(before,'support_bars[0]',{path:np.ones(3)})
        before["support_bars[0]['database'].D13"][0,0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'Invalid captured table'):
            check.active_table_points(before,2)


if __name__ == '__main__':
    unittest.main()
