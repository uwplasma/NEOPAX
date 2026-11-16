"""Saved-input and CPU-refusal checks; no real JAX import or physics solve."""
from contextlib import redirect_stderr
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("field_sensitivity",
    ROOT/"examples/benchmarks/check_saved_ntx_field_sensitivity.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class SavedNTXFieldTests(unittest.TestCase):
    def test_reported_field_values_remain_one_ulp_apart(self):
        before = np.float64(float.fromhex("0x1.c035ba9ccdac0p-7"))
        after = np.nextafter(before, -np.inf)
        values = {"scan_inputs['channels'].dr_tildeds": np.array([before, after]),
                  "scan_inputs['channels'].b00": np.ones(2)}
        self.assertEqual(check.field_at(values, [1.], 0, 0).hex(), "0x1.c035ba9ccdac0p-7")
        self.assertEqual(check.field_at(values, [1.], 1, 0).hex(), "0x1.c035ba9ccdabfp-7")
        self.assertEqual(float(before-after), 1.734723475976807e-18)

    def test_restore_only_saved_surface_fields_and_original_static_metadata(self):
        prefix = "scan_inputs['surfaces'][6]."
        fields = {"b_cos": np.array([9., .1]), "m": np.array([0,1]),
            "n": np.array([0,0]), "nfp": np.array(2), "requested_psi_n": np.array(.875**2),
            "psi_n": np.array(.875**2), "iota": np.array(-.6)}
        arrays = {prefix+k: v for k,v in fields.items()}
        arrays["scan_inputs['surfaces'][5].b_cos"] = np.ones(3)
        context = SimpleNamespace(input_path=Path("original-input"),
            static=SimpleNamespace(resolution=SimpleNamespace(ns=21, mpol=4, ntor=3)))
        loaded = check.surface_fields(arrays, 6)
        result = check.restore_surface(loaded, context, SimpleNamespace, np.asarray)
        np.testing.assert_array_equal(result.b_cos, fields["b_cos"])
        self.assertEqual(result.path, context.input_path)
        self.assertEqual((result.ns, result.mpol, result.ntor, result.nfp), (21,4,3,2))
        self.assertEqual((result.total_mode_count, result.loaded_mode_count), (2,2))
        self.assertIsInstance(result.requested_psi_n, float)

    def test_cpu_backend_is_rejected_before_importing_physics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/"config.toml").write_text('[neoclassical]\nflux_model="ntx_scan_runtime"\n')
            for name in ("ad", "fd_2"):
                (root/name).mkdir()
                np.savez(root/name/"generic_database_baseline_test.npz",
                    **{"scan_inputs['surfaces'][0].b_cos": np.ones(2)})
            with patch.object(sys, "argv", [str(check.__file__), "--run-dir", str(root),
                    "--out-dir", str(root/"unused")]), \
                 patch.object(check, "POINTS", ((0,0,0),)), \
                 patch.dict(sys.modules, jax=SimpleNamespace(default_backend=lambda: "cpu")), \
                 redirect_stderr(io.StringIO()) as errors:
                with self.assertRaises(SystemExit):
                    check.main()
                self.assertIn("no CPU fallback", errors.getvalue())
            self.assertFalse((root/"unused").exists())
            self.assertNotIn("jax", sys.modules)


if __name__ == "__main__":
    unittest.main()
