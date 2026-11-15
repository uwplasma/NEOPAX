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
    def test_original_b0_maximum_is_retained(self):
        source = path.parents[2] / "NEOPAX/_geometry_autodiff.py"
        builder = next(node for node in ast.parse(source.read_text()).body
                       if isinstance(node, ast.FunctionDef)
                       and node.name == "_traceable_vmec_surface_from_tables")
        expression = next(keyword.value for node in ast.walk(builder) if isinstance(node, ast.Call)
                          for keyword in node.keywords if keyword.arg == "b0")
        self.assertEqual(ast.dump(expression), ast.dump(ast.parse("jnp.max(jnp.abs(bmnc))", mode="eval").body))

    def test_empty_new_parameter_is_ignored_but_nonempty_is_rejected(self):
        captured = {"params.rbc": np.ones(2), "params.ac_aux_f": np.empty(0)}
        self.assertEqual(list(check.parameter_arrays(captured, ("rbc",))), ["rbc"])
        captured["params.ac_aux_f"] = np.ones(1)
        with self.assertRaisesRegex(ValueError, "cannot represent nonempty"):
            check.parameter_arrays(captured, ("rbc",))

    def test_reference_field_report_reads_each_parameter_and_radius(self):
        tangents = {"payload_paths[0]": "['surfaces'][0].b0",
                    "payload_paths[1]": "['surfaces'][0].b_cos",
                    "payload_tangents[0]": np.array([0., -8.]),
                    "payload_tangents[1]": np.array([[-11., 2.], [-8., 1.]]),
                    "parameter_labels[0]": "RBC:1:0", "parameter_labels[1]": "ZBS:1:0"}
        rows = check.surface_b0_rows(tangents, [.5])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["b0_tangent"], 0.)
        self.assertEqual(rows[0]["coefficient0_tangent"], -11.)
        self.assertEqual(rows[1]["parameter"], "ZBS:1:0")

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
