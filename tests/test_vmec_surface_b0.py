"""Check the production surface reference-field expression in isolation."""
import ast
from pathlib import Path
import unittest

import numpy as np


source = Path(__file__).resolve().parents[1] / "NEOPAX/_geometry_autodiff.py"
builder = next(node for node in ast.parse(source.read_text()).body
               if isinstance(node, ast.FunctionDef)
               and node.name == "_traceable_vmec_surface_from_tables")
expression = next(keyword.value for node in ast.walk(builder) if isinstance(node, ast.Call)
                  for keyword in node.keywords if keyword.arg == "b0")
compiled_expression = compile(ast.Expression(expression), str(source), "eval")


def reference_field(coefficients, array_module):
    return eval(compiled_expression, {"bmnc": coefficients, "jnp": array_module})


class SurfaceReferenceFieldTests(unittest.TestCase):
    def test_numpy_forward_value_matches_maximum_including_ties_and_zero(self):
        for values in ([8.519623583796868, -1.823213973183598, .01],
                       [1., -9., 2.], [-4., 1., 2.], [2., -2., 1.], [0., 0., 0.]):
            with self.subTest(values=values):
                x = np.asarray(values)
                self.assertEqual(reference_field(x, np), np.max(np.abs(x)))

    def test_jvp_vjp_and_jit_follow_the_maximizing_coefficient(self):
        # This imports JAX only when explicitly running this numerical test.
        # It uses three coefficients; no VMEX, NTX or transport computation.
        import jax
        import jax.numpy as jnp

        enable_x64 = getattr(jax, "enable_x64", None)
        if enable_x64 is None:
            from jax.experimental import enable_x64
        with enable_x64():
            cases = (
                ([8.519623583796868, -1.823213973183598, .01],
                 [-11.471163375509532, 2.1483627659136384, .1]),
                ([1., -9., 2.], [.2, -.7, .3]),
                ([-4., 1., 2.], [.6, .4, -.1]),
            )
            for values, direction in cases:
                x, dx = jnp.asarray(values), jnp.asarray(direction)
                index = int(np.argmax(np.abs(values)))
                expected_gradient = np.zeros(3)
                expected_gradient[index] = np.sign(values[index])
                expected_direction = float(expected_gradient @ np.asarray(direction))
                for fn in (lambda v: reference_field(v, jnp),
                           jax.jit(lambda v: reference_field(v, jnp))):
                    with self.subTest(values=values, compiled=hasattr(fn, "lower")):
                        value, tangent = jax.jvp(fn, (x,), (dx,))
                        _, pullback = jax.vjp(fn, x)
                        np.testing.assert_array_equal(value, np.max(np.abs(values)))
                        np.testing.assert_allclose(tangent, expected_direction, rtol=1e-14, atol=1e-14)
                        np.testing.assert_array_equal(pullback(jnp.asarray(1.))[0], expected_gradient)
                        h = 1e-6
                        fd = float((fn(x+h*dx)-fn(x-h*dx))/(2*h))
                        self.assertAlmostEqual(fd, expected_direction, delta=2e-9)
            x = jnp.asarray([2., -2., 1.])
            np.testing.assert_array_equal(
                jax.grad(lambda v: reference_field(v, jnp))(x), [1., 0., 0.])


if __name__ == "__main__":
    unittest.main()
