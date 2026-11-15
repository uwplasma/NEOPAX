"""Saved-array bookkeeping only. Import no JAX or physics code."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

path = Path(__file__).resolve().parents[1] / "examples/benchmarks/check_saved_compact_geometry_gradient.py"
spec = importlib.util.spec_from_file_location("saved_compact_check", path)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class SavedCompactGradientTests(unittest.TestCase):
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
