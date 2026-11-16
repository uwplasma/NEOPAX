"""Saved-input and CPU-refusal checks; no real JAX import or physics solve."""
from contextlib import redirect_stderr
from dataclasses import dataclass
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
    def test_aot_lowering_receives_flat_leaves_and_keeps_inputs_dynamic(self):
        @dataclass
        class Surface:
            m: object
            scale: object
            def __post_init__(self):
                len(self.m)

        class ArgInfo:
            pass

        def rebuild(_, values):
            return Surface(*values[:2]), *values[2:]

        def jit(function):
            def lower(*args):
                for arg in args:
                    if isinstance(arg, Surface):
                        Surface(ArgInfo(), ArgInfo())  # Reproduce the old AOT metadata failure.
                return SimpleNamespace(compile=lambda: function)
            return SimpleNamespace(lower=lower)

        backend = SimpleNamespace(jit=jit, tree_util=SimpleNamespace(tree_unflatten=rebuild))
        function = lambda surface, nu, es: surface.scale*np.sum(surface.m)+nu+es
        with self.assertRaisesRegex(TypeError, "ArgInfo"):
            jit(function).lower(Surface(np.array([1.,2.]), 3.), .1, .2)
        leaves = (np.array([1.,2.]), 3., .1, .2)
        compiled = check.compile_flat_inputs(function, leaves, object(), backend)
        self.assertAlmostEqual(compiled(*leaves), 9.3)
        self.assertAlmostEqual(compiled(*leaves[:-1], .4), 9.5)
        self.assertAlmostEqual(compiled(np.array([2.,3.]), *leaves[1:]), 15.3)

    def test_reduced_head_matches_full_nonsymmetric_block_system(self):
        rng = np.random.default_rng(7)
        n, size = 5, 2
        diagonal = 4*np.tile(np.eye(size), (n,1,1))+rng.normal(size=(n,size,size))*.1
        lower, upper = (rng.normal(size=(n,size,size))*.2 for _ in range(2))
        matrix = np.zeros((n*size,n*size))
        for j in range(n):
            matrix[j*size:(j+1)*size,j*size:(j+1)*size] = diagonal[j]
            if j:
                matrix[j*size:(j+1)*size,(j-1)*size:j*size] = lower[j]
            if j+1 < n:
                matrix[j*size:(j+1)*size,(j+1)*size:(j+2)*size] = upper[j]
        rhs = np.zeros((n*size,2))
        rhs[:3*size] = rng.normal(size=(3*size,2))
        tail = diagonal[-1]
        for j in range(n-2,1,-1):
            tail = diagonal[j]-upper[j] @ np.linalg.solve(tail, lower[j+1])
        head = check.head_matrix(diagonal, lower, upper, tail, np)
        np.testing.assert_allclose(np.linalg.solve(head,rhs[:3*size]),
            np.linalg.solve(matrix,rhs)[:3*size], rtol=1e-13, atol=1e-14)

    def test_two_sided_scaling_preserves_original_equations(self):
        matrix = np.diag([2.**-30,2.**30]) @ np.array([[4.,1.],[2.,3.]]) @ np.diag([2.**4,2.**-4])
        expected = np.array([[.5,1.],[-.25,.5]])
        rhs = matrix @ expected
        actual, scaled = check.scaled_solve(matrix,rhs,np)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(matrix @ actual, rhs, rtol=1e-14, atol=0.)
        self.assertLess(np.max(np.abs(scaled))/np.min(np.abs(scaled)),
                        np.max(np.abs(matrix))/np.min(np.abs(matrix)))

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
