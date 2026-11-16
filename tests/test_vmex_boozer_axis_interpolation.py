"""Exercise the actual adapter's interpolation kernel with NumPy; no JAX import."""
import ast
from pathlib import Path
import unittest

import numpy as np


def kernel(s, rows, modes):
    path = Path(__file__).resolve().parents[1]/'NEOPAX/_geometry_autodiff.py'
    tree = ast.parse(path.read_text())
    outer = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
        and n.name == '_neopax_vmex_boozer_table_inputs_from_state')
    function = next(n for n in outer.body if isinstance(n, ast.FunctionDef) and n.name == 'spectral_half')
    sqrt_s = np.sqrt(s)
    namespace = dict(jnp=np, rows=rows, xm=modes, sqrt_s=sqrt_s,
        s_half=.5*(s[rows]+s[rows-1]), project=lambda x, _: x, mirror=lambda x, _: x,
        phys_rows=lambda even, odd, i: even[i]+sqrt_s[i,None]*odd[i])
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['spectral_half']


class BoozerAxisInterpolationTests(unittest.TestCase):
    def test_axis_projection_roundoff_is_not_amplified(self):
        s = np.array([0., .25, 1.])
        modes = np.array([0, 1, 2, 3])
        values = np.array([[10., 8e-17, 5., -2e-17], [12., 2., 6., .25], [14., 4., 7., 1.]])
        result = kernel(s, np.array([1,2]), modes)(values, np.zeros_like(values), 'even')
        # The VMEX/booz_xform convention divides the axis coefficient by one.
        np.testing.assert_allclose(result[0], [11., np.sqrt(.125)*2., 5.5, np.sqrt(.125)*.25], rtol=1e-15, atol=1e-16)
        self.assertLess(np.max(np.abs(result)), 20.)

    def test_even_modes_and_nonaxis_odd_modes_retain_their_radial_interpolation(self):
        s = np.array([0., .1, .4, 1.])
        rows = np.array([2,3])
        modes = np.array([0, 1, 2, 3])
        values = np.column_stack((5+6*s, (3+4*s)*np.sqrt(s), 2*s-1, (-2+3*s)*np.sqrt(s)))
        half = .5*(s[rows]+s[rows-1])
        expected = np.column_stack((5+6*half, (3+4*half)*np.sqrt(half), 2*half-1, (-2+3*half)*np.sqrt(half)))
        result = kernel(s, rows, modes)(values, np.zeros_like(values), 'even')
        np.testing.assert_allclose(result, expected, rtol=1e-15, atol=1e-15)

    def test_axis_coefficient_sensitivity_is_finite_and_linear(self):
        s = np.array([0., .25, 1.])
        evaluate = kernel(s, np.array([1,2]), np.array([1]))
        direction = np.array([[1.], [0.], [0.]])
        zero = np.zeros_like(direction)
        expected = np.array([[.5*np.sqrt(.125)], [0.]])
        np.testing.assert_allclose(evaluate(direction, zero, 'even'), expected, rtol=1e-15)
        h = 1e-5
        difference = (evaluate(h*direction, zero, 'even')-evaluate(-h*direction, zero, 'even'))/(2*h)
        np.testing.assert_allclose(difference, expected, rtol=1e-15)


if __name__ == '__main__':
    unittest.main()
