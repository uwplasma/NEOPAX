"""Saved-state and compiler-observer contracts; no JAX or physics imports."""
import dataclasses
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples/benchmarks'))
from check_saved_geometry_build import aligned_trace, capture_geometry_ir, equilibrium_context, saved_equilibrium


class SavedGeometryBuildTests(unittest.TestCase):
    def test_original_history_control_does_not_replace_the_configured_equilibrium_solve(self):
        solve, callback, restore = (lambda: 'solve'), (lambda: 'callback'), (lambda: 'recorded')
        implicit = SimpleNamespace(_host_solve=solve, _host_solve_and_mask_impl=callback)
        with equilibrium_context(implicit, restore, fresh=True):
            self.assertIs(implicit._host_solve, solve)
            self.assertIs(implicit._host_solve_and_mask_impl, callback)
        with equilibrium_context(implicit, restore, fresh=False):
            self.assertIs(implicit._host_solve_and_mask_impl, restore)
            with self.assertRaisesRegex(AssertionError, 'disabled'):
                implicit._host_solve()
        self.assertIs(implicit._host_solve, solve)
        self.assertIs(implicit._host_solve_and_mask_impl, callback)

    def test_callback_restores_exact_arrays_and_rejects_changed_parameters(self):
        @dataclasses.dataclass
        class Params:
            boundary: object
        array = np.array([1., -0., 3.])
        capture = {'state.R': array, 'mask.R': np.ones(3), 'params.boundary': array.copy()}
        restore = saved_equilibrium(capture, SimpleNamespace)
        state, mask = restore(None, Params(array.copy()))
        self.assertIs(state.R, array)
        np.testing.assert_array_equal(mask.R, np.ones(3))
        changed = array.copy()
        changed[1] = 0.
        with self.assertRaisesRegex(ValueError, 'parameters differ'):
            restore(None, Params(changed))
        self.assertEqual(capture['state.R'].tobytes(), array.tobytes())

    def test_observer_keeps_the_original_compile_call_and_executable(self):
        calls = []
        executable = SimpleNamespace(hlo_modules=lambda: [SimpleNamespace(to_string=lambda: 'optimized')])
        def compile_original(*args, **kwargs):
            calls.append((args, kwargs))
            return executable
        compiler = SimpleNamespace(compile_or_get_cached=compile_original)
        class Computation:
            operation = SimpleNamespace(attributes={'sym_name': 'jit_booz_xform_jax_impl'})
            def __str__(self):
                return 'original stablehlo'
        computation, options = Computation(), object()
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            with capture_geometry_ir(directory, compiler):
                actual = compiler.compile_or_get_cached('backend', computation, options=options)
                self.assertIs(actual, executable)
            self.assertIs(compiler.compile_or_get_cached, compile_original)
            self.assertEqual(calls, [(('backend', computation), {'options': options})])
            self.assertEqual((directory/'program_000.mlir').read_text(), str(computation))
            self.assertEqual((directory/'program_000.txt').read_text(), 'optimized')
            self.assertEqual(len(json.loads((directory/'programs.json').read_text())), 1)

    def test_trace_alignment_retains_repeated_boundaries(self):
        a = {"trace['00_vmex_geometry']['R']": np.array([1.]),
             "trace['04_vmex_geometry']['R']": np.array([2.]),
             "trace['05_geometry']['R0']": np.array(3.)}
        b = {"trace['03_vmex_geometry']['R']": np.array([1.]),
             "trace['07_vmex_geometry']['R']": np.array([2.]),
             "trace['08_geometry']['R0']": np.array(3.)}
        left, right = aligned_trace(a), aligned_trace(b)
        self.assertEqual(set(left), set(right))
        self.assertEqual(len(left), 3)
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])


if __name__ == '__main__':
    unittest.main()
