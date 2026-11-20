"""Saved-array bookkeeping only. Import no JAX or physics code."""
import ast
import importlib.util
import inspect
from pathlib import Path
import unittest

import numpy as np

path = Path(__file__).resolve().parents[1] / "examples/benchmarks/check_saved_compact_geometry_gradient.py"
spec = importlib.util.spec_from_file_location("saved_compact_check", path)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class SavedCompactGradientTests(unittest.TestCase):
    def test_tangent_effect_is_quantified_for_every_cost_with_replay_control(self):
        captured = {'parameter_labels[0]':np.asarray('ZBS:1:0'),
            'geometry_gradients':np.asarray([[3.],[10.]])}
        report = {'parameter_name':'ZBS:1:0','objective_labels':['Er','bootstrap'],'gradient_fd':[4.,12.]}
        rows = check.gradient_direction_rows(captured,'ZBS:1:0',[[3.],[10.]],[[4.],[12.]],report,['Er','bootstrap'])
        self.assertEqual([r['tangent_effect'] for r in rows],[1.,2.])
        self.assertEqual([r['control_relative_difference'] for r in rows],[0.,0.])
        self.assertEqual([r['projected_fd_relative_difference'] for r in rows],[0.,0.])
        # A changed payload replay is exposed independently of tangent effects.
        rows = check.gradient_direction_rows(captured,'ZBS:1:0',[[3.3],[10.]],[[4.],[12.]],report,['Er','bootstrap'])
        self.assertAlmostEqual(rows[0]['control_relative_difference'],.1)
        with self.assertRaises(ValueError):
            check.gradient_direction_rows(captured,'ZBS:1:0',[[3.],[10.]],[[np.nan],[12.]],report,['Er','bootstrap'])
        with self.assertRaises(ValueError):
            check.gradient_direction_rows(captured,'ZBS:1:0',[[3.],[10.]],[[4.],[12.]],report,['bootstrap','Er'])

    def test_fd_tangent_comparison_selects_parameter_and_preserves_mismatch(self):
        captured = {'parameter_labels[0]':np.asarray('RBC:1:0'),
            'parameter_labels[1]':np.asarray('ZBS:1:0'),
            'state_tangents.R_cos':np.asarray([[7.,8.],[3.,4.]]),
            'state_tangents.Z_sin':np.zeros((2,2))}
        rows,reference = check.state_tangent_rows(captured,'ZBS:1:0',
            {'R_cos':np.asarray([3.,4.5]),'Z_sin':np.zeros(2)})
        self.assertEqual(rows[0]['different_entries'],1)
        self.assertEqual(rows[0]['max_abs_difference'],.5)
        self.assertAlmostEqual(rows[0]['relative_l2_difference'],.1)
        self.assertEqual(rows[1]['relative_l2_difference'],0.)
        np.testing.assert_array_equal(reference['R_cos'],[3.,4.])
        with self.assertRaises(ValueError):
            check.state_tangent_rows(captured,'missing',{'R_cos':np.zeros(2)})
        for current in ({'R_cos':np.zeros(2)},
                        {'R_cos':np.ones(3),'Z_sin':np.zeros(2)},
                        {'R_cos':np.asarray([np.nan,4.]),'Z_sin':np.zeros(2)}):
            with self.assertRaises(ValueError):
                check.state_tangent_rows(captured,'ZBS:1:0',current)

    def test_production_call_supplies_the_actual_required_arguments_and_configured_grid(self):
        name = "geometry_payload_pullback_from_param_vector_raw_block_transpose"
        source = path.parents[2] / "NEOPAX/_geometry_autodiff.py"
        definition = next(node for node in ast.parse(source.read_text()).body
                          if isinstance(node, ast.FunctionDef) and node.name == name)
        parameters = [inspect.Parameter(arg.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                      for arg in definition.args.args]
        parameters.extend(inspect.Parameter(arg.arg, inspect.Parameter.KEYWORD_ONLY,
                                            default=inspect.Parameter.empty if default is None else None)
                          for arg, default in zip(definition.args.kwonlyargs,
                                                  definition.args.kw_defaults, strict=True))
        call = next(node for node in ast.walk(ast.parse(path.read_text()))
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == name)
        keywords = {kw.arg: kw.value for kw in call.keywords}
        inspect.Signature(parameters).bind(*[None] * len(call.args),
                                           **dict.fromkeys(keywords))
        config = {"neoclassical": {"ntx_scan_n_theta": 7, "ntx_scan_n_zeta": 9,
                                   "ntx_scan_n_xi": 11}}
        for key, expected in (("n_theta", 7), ("n_zeta", 9), ("n_xi", 11)):
            value = eval(compile(ast.Expression(keywords[key]), str(path), "eval"),
                         {"config": config})
            self.assertEqual(value, expected)

    def test_flat_field_prefix_does_not_capture_siblings_or_nested_fields(self):
        arrays = {"params.rbc": np.ones(2), "params_extra.rbc": np.zeros(2),
                  "params.nested.value": np.zeros(2)}
        self.assertEqual(list(check.named_arrays(arrays, "params.")), ["rbc"])

    def test_all_objective_and_parameter_columns_are_contracted_by_named_path(self):
        captured = {"parameter_labels[0]": "RBC:1:0", "parameter_labels[1]": "ZBS:1:0"}
        path = "['geometry'].Bsqav"
        captured["payload_bars[0]" + path] = np.array([2., -1.])
        captured["payload_bars[1]" + path] = np.array([-.5, 3.])
        tangents = {"payload_paths[0]": path, "payload_tangents[0]": np.array([[1., 2.], [3., 4.]])}
        current = np.array([[0., 2.], [5.5, 10.5]])
        rows, reconstructed = check.contraction_rows(captured, tangents, current)
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[-1]["parameter"], "ZBS:1:0")
        self.assertEqual(rows[-1]["contribution"], 10.5)
        np.testing.assert_array_equal(reconstructed, current)
        # Preserve a disagreement for diagnosis rather than dropping the rows.
        different_rows, same_projection = check.contraction_rows(captured, tangents, current + .01)
        self.assertEqual(different_rows, rows)
        np.testing.assert_array_equal(same_projection, reconstructed)
        tangents["payload_tangents[0]"] = np.zeros((2, 3))
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            check.contraction_rows(captured, tangents, current)


if __name__ == "__main__":
    unittest.main()
