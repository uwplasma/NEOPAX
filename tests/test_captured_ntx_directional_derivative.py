"""Saved-array preparation and API contracts; import no physics packages."""
import ast
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1]/'examples/benchmarks/check_captured_ntx_directional_derivative.py'
JAX_ALREADY_LOADED = 'jax' in sys.modules
spec = importlib.util.spec_from_file_location('captured_ntx_check', SCRIPT)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class CapturedNTXPreparationTests(unittest.TestCase):
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
        self.assertEqual('jax' in sys.modules, JAX_ALREADY_LOADED)
        tree = ast.parse(SCRIPT.read_text())
        calls = {node.func.attr for node in ast.walk(tree)
                 if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)}
        self.assertIn('solve_prepared_coefficient_vector', calls)
        self.assertIn('pullback_prepared_database_coefficients_to_vmec_primitive_multi_rhs', calls)
        names = {node.func.id for node in ast.walk(tree)
                 if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)}
        self.assertIn('_prepared_implicit_vjp_primal', names)
        self.assertEqual(check.COEFFICIENTS, (('D11',0),('D13',2),('D33',3)))


if __name__ == '__main__':
    unittest.main()
