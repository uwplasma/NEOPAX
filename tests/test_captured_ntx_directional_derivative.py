"""Saved-array preparation and API contracts; import no physics packages."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1]/'examples/benchmarks/check_captured_ntx_directional_derivative.py'
JAX_ALREADY_LOADED = 'jax' in sys.modules
spec = importlib.util.spec_from_file_location('captured_ntx_check', SCRIPT)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)
JAX_AFTER_IMPORT = 'jax' in sys.modules


class CapturedNTXPreparationTests(unittest.TestCase):
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
