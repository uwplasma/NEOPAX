"""Saved-array diagnostics only: these tests never import JAX or run physics."""
import dataclasses
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "examples/benchmarks/check_saved_transport_baseline_repeatability.py"
spec = importlib.util.spec_from_file_location("baseline_repeat_check", SCRIPT)
check = importlib.util.module_from_spec(spec)
_jax_before = "jax" in sys.modules
spec.loader.exec_module(check)
_jax_after = "jax" in sys.modules


class SavedBaselineTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
