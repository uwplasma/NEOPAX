"""Forward precision regressions; NumPy/stdlib only, no JAX import."""
from __future__ import annotations

import ast
import contextlib
import dataclasses
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT/'examples/benchmarks/_fd_native_precision.py'
spec = importlib.util.spec_from_file_location('fd_native_precision', FILE)
precision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(precision)


@dataclasses.dataclass
class State:
    x: object


@dataclasses.dataclass
class Surface:
    value: object
    modes: object
    optional: object = None


class NativePrecisionTests(unittest.TestCase):
    def test_nested_comprehensions_cannot_call_original_array_backend(self):
        legacy = Mock(side_effect=TypeError('float128 reached the original JAX backend'))
        namespace = {'jnp':SimpleNamespace(concatenate=legacy)}
        exec('''
def stage(x):
    return jnp.concatenate((x, x), axis=-1)
def forward(rows):
    def nested():
        return [stage(row) for row in rows]
    return tuple(value for value in nested())
''', namespace)
        copy = precision.extended_forward(namespace['forward'], lambda name: None)
        value = np.asarray([[1., 2.], [3., 4.]], dtype=np.longdouble)
        output = copy(value)
        np.testing.assert_array_equal(output, [[1.,2.,1.,2.],[3.,4.,3.,4.]])
        self.assertTrue(all(row.dtype == np.longdouble for row in output))
        self.assertIsNot(copy.__globals__['stage'], namespace['stage'])
        self.assertIs(namespace['jnp'].concatenate, legacy)
        legacy.assert_not_called()

    def test_original_code_runs_with_independent_extended_backend(self):
        namespace = {'jnp': np}
        exec('''
def square(x):
    return jnp.asarray(x, dtype=jnp.float64)**2
def forward(x):
    a = square(x)
    a = a.at[0].set(a[0] + 2)
    a = a.at[1:].multiply(3)
    return jnp.einsum('i,i->', a, jnp.ones_like(a), precision='highest')
''', namespace)
        copy = precision.extended_forward(namespace['forward'], lambda name: None)
        x = np.asarray([12.048933794228684, 4.142495110048192], dtype=np.longdouble)
        direction = np.asarray([-.06278770474858161, -.04522143496511888], dtype=np.longdouble)
        h = np.longdouble('3.3946185202503675e-7')
        fd = (copy(x+h*direction) - copy(x-h*direction))/(2*h)
        expected = 2*x[0]*direction[0] + 6*x[1]*direction[1]
        self.assertLess(abs((fd-expected)/expected), 1e-10)
        self.assertIs(namespace['jnp'], np)
        self.assertIsNot(copy.__globals__['square'], namespace['square'])
        np.testing.assert_array_equal(x, [np.longdouble(12.048933794228684), np.longdouble(4.142495110048192)])

    def test_cubic_matches_independent_linear_and_interior_quadratic(self):
        x = np.asarray([0., 1., 2., 3.], dtype=np.longdouble)
        for q in (-.25, .5, 1.25, 3.2):
            result = precision._cubic(q, x, np.stack((2*x+3, 7*x-2), axis=-1), method='cubic', extrap=True)
            np.testing.assert_allclose(result, [2*q+3, 7*q-2], rtol=1e-15)
        self.assertAlmostEqual(float(precision._cubic(1.25, x, x*x, method='cubic', extrap=True)), 1.25**2)
        with self.assertRaisesRegex(ValueError, 'cubic/extrap'):
            precision._cubic(1., x, x, method='linear', extrap=True)

    def test_baseline_is_untouched_and_displacement_survives_rounding(self):
        normal = Mock(side_effect=lambda ctx, state, **kw: (Surface(np.asarray(100.), np.array([0, 1])),))
        host = Mock(side_effect=lambda ctx, state, **kw: (Surface(state.x.copy(), np.array([0, 1])),))
        module = SimpleNamespace(_traceable_vmec_surfaces_from_state=normal, _import_vmec_module=Mock())
        # The tiny displacement would disappear if added to 1 in float64.
        state, direction = State(np.asarray(1.)), State(np.asarray(1.))
        with patch.object(precision, 'extended_forward', return_value=host):
            evaluator = precision.NativePrecision(module, state, direction)
        a = evaluator.evaluate(None, 0., s_values=(.5,))
        self.assertEqual(a[0].value.tobytes(), np.asarray(100.).tobytes())
        evaluator.evaluate(None, 1e-17, s_values=(.5,))
        intended = host.call_args.args[1].x
        self.assertGreater(intended, np.longdouble(1.))
        self.assertEqual(normal.call_count, 1)
        normal.assert_called_once_with(None, state, s_values=(.5,))
        self.assertEqual(state.x, 1.)
        self.assertIsNone(a[0].optional)
        np.testing.assert_array_equal(a[0].modes, [0, 1])
        for bad in (np.asarray(np.nan), np.asarray([1., 2.])):
            with self.assertRaisesRegex(ValueError, 'Invalid extended native field'):
                precision.recenter(a[0], Surface(bad, []), Surface(np.asarray(1.), []))

    def test_runtime_hook_keeps_zero_step_and_float64_path_unchanged(self):
        file = ROOT/'examples/benchmarks/benchmark_transport_realtime_geometry_forward_fd.py'
        fn = next(n for n in ast.parse(file.read_text()).body if isinstance(n, ast.FunctionDef)
                  and n.name == '_runtime_for_frozen_linearized_geometry_step')
        original = Mock()
        geometry = SimpleNamespace(_traceable_vmec_surfaces_from_state=original)
        output = object()
        def build(config, context, state, **kwargs):
            if config.get('read_surface'):
                return geometry._traceable_vmec_surfaces_from_state(context, state, s_values=(.5,)), output
            return 'ordinary', output
        namespace = dict(contextlib=contextlib, jnp=np, dataclasses=dataclasses,
            jax=SimpleNamespace(tree=SimpleNamespace(map=lambda fn,x,d: State(fn(x.x,d.x)))),
            build_runtime_context_for_vmec_state=Mock(side_effect=build))
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(file), 'exec'), namespace)
        run = namespace[fn.name]
        bundle = dict(state_star=State(np.asarray(1.)), state_tangent=State(np.asarray(2.)),
                      context=object(), n_r=51, native_precision='extended')
        factory = Mock(return_value=SimpleNamespace(evaluate=Mock(return_value='corrected'),
            original=original, remember=Mock()))
        modules = {'NEOPAX':SimpleNamespace(_geometry_autodiff=geometry),
                   '_fd_native_precision':SimpleNamespace(NativePrecision=factory)}
        with patch.dict(sys.modules, modules):
            self.assertEqual(run({}, bundle, step_scale=0.), ('ordinary', output))
            factory.return_value.evaluate.assert_not_called()
            original.return_value='untouched baseline'
            self.assertEqual(run({'read_surface':True}, bundle, step_scale=0.), ('untouched baseline', output))
            factory.return_value.remember.assert_called_once_with((.5,), 'untouched baseline')
            self.assertEqual(run({'read_surface':True}, bundle, step_scale=1e-7), ('corrected', output))
            run({'read_surface':True}, bundle, step_scale=-1e-7)
            factory.assert_called_once()
            self.assertIs(geometry._traceable_vmec_surfaces_from_state, original)
            bundle['native_precision']='float64'
            self.assertEqual(run({}, bundle, step_scale=1e-7), ('ordinary', output))
        self.assertNotIn('jax', sys.modules)


if __name__ == '__main__':
    unittest.main()
