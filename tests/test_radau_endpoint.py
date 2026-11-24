"""Endpoint arithmetic checks; CUDA checks are opt-in and run no transport solve."""
import ast
import os
from pathlib import Path
import time
from types import SimpleNamespace
import unittest

import numpy as np


SOURCE = Path(__file__).resolve().parents[1]/"NEOPAX/_transport_solvers.py"


def production_endpoint(jax_module, array_module):
    # Load the actual function without initializing VMEX/NTX or the package.
    functions = [node for node in ast.parse(SOURCE.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in
                 {"_radau_weighted_stage_sum", "_radau_step_endpoint"}]
    namespace = dict(jax=jax_module, jnp=array_module)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["_radau_step_endpoint"]


class RadauEndpointNumpyTests(unittest.TestCase):
    def test_stage_counts_and_precisions_retain_the_radau_equation(self):
        endpoint = production_endpoint(SimpleNamespace(lax=SimpleNamespace(optimization_barrier=lambda x:x)), np)
        rng = np.random.default_rng(502)
        for dtype in (np.float32, np.float64):
            for count in (1, 2, 3, 5, 7):
                with self.subTest(dtype=dtype, stages=count):
                    y = rng.normal(size=408).astype(dtype)
                    weights = rng.normal(size=count).astype(dtype)
                    stages = rng.normal(size=(count, 408)).astype(dtype)
                    h = np.asarray(.013, dtype=dtype)
                    actual = endpoint(y, h, weights, stages)
                    terms = weights.astype(np.longdouble)[:,None]*stages.astype(np.longdouble)
                    expected = y.astype(np.longdouble)+np.longdouble(h)*np.sum(terms, axis=0)
                    bound = 16*np.finfo(dtype).eps*(np.abs(y)+abs(h)*np.sum(np.abs(terms), axis=0))
                    self.assertEqual(actual.dtype, dtype)
                    self.assertTrue(np.all(np.abs(actual-expected) <= bound))
                    np.testing.assert_array_equal(endpoint(y, np.asarray(0., dtype=dtype), weights, stages), y)

    def test_storage_layout_and_strides_do_not_change_endpoint_bits(self):
        endpoint = production_endpoint(SimpleNamespace(lax=SimpleNamespace(optimization_barrier=lambda x:x)), np)
        rng = np.random.default_rng(71)
        stages = rng.normal(size=(7, 408))
        weights, y = rng.normal(size=7), rng.normal(size=408)
        expected = endpoint(y, .001, weights, stages)
        padded = np.zeros((7, 816))
        padded[:,::2] = stages
        for other in (np.asfortranarray(stages), padded[:,::2], stages[:,::-1][:,::-1]):
            self.assertEqual(endpoint(y, .001, weights, other).tobytes(), expected.tobytes())


@unittest.skipUnless(os.environ.get("NEOPAX_TEST_RADAU_ENDPOINT_GPU") == "1", "opt-in CUDA check")
class RadauEndpointCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as jnp
        jax.config.update("jax_enable_x64", True)
        if jax.default_backend() != "gpu":
            raise RuntimeError("This check requires a CUDA GPU.")
        cls.jax, cls.jnp = jax, jnp
        cls.endpoint = staticmethod(production_endpoint(jax, jnp))

    def arrays(self, count=7):
        rng = np.random.default_rng(71)
        return tuple(self.jnp.asarray(x) for x in (
            rng.normal(size=408), np.asarray(.013), rng.normal(size=count), rng.normal(size=(count, 408))))

    def test_endpoint_bits_match_across_jit_layout_and_batched_contexts(self):
        jax, jnp, endpoint = self.jax, self.jnp, self.endpoint
        values = self.arrays()
        expected = np.asarray(jax.jit(endpoint)(*values))
        auxiliary = jax.jit(lambda y,h,w,k: (endpoint(y,h,w,k), h*((w-.1) @ k)))
        nested = jax.jit(lambda y,h,w,k: jax.jit(endpoint)(y,h,w,k))
        transposed = jax.jit(lambda y,h,w,k: endpoint(y,h,w,k.T))
        batched = jax.jit(jax.vmap(endpoint, in_axes=(None,None,None,0)))
        results = [auxiliary(*values)[0], nested(*values),
                   transposed(*values[:3], values[3].T)]
        results.extend(batched(*values[:3], jnp.stack([values[3]]*8)))
        for actual in results:
            self.assertEqual(np.asarray(actual).tobytes(), expected.tobytes())
        # The previous failure was autotuning a generic stage dot/reduction.
        ir = jax.jit(endpoint).lower(*values).as_text()
        self.assertNotIn("stablehlo.dot_general", ir)
        self.assertNotIn("stablehlo.reduce", ir)

    def test_state_time_weight_stage_jvp_vjp_and_second_derivative(self):
        jax, jnp, endpoint = self.jax, self.jnp, self.endpoint
        rng = np.random.default_rng(8)
        for count in (2, 3, 5, 7):
            values = self.arrays(count)
            y,h,w,k = values
            dots = tuple(jnp.asarray(rng.normal(size=x.shape)) for x in values)
            dy,dh,dw,dk = dots
            value, tangent = jax.jit(lambda p,d: jax.jvp(endpoint,p,d))(values,dots)
            np.testing.assert_allclose(value, y+h*(w @ k), rtol=2e-13, atol=2e-13)
            np.testing.assert_allclose(tangent, dy+dh*(w @ k)+h*(dw @ k+w @ dk), rtol=2e-13, atol=2e-13)
            bar = jnp.asarray(rng.normal(size=y.shape))
            bars = jax.jit(lambda p,b: jax.vjp(endpoint,*p)[1](b))(values,bar)
            expected = (bar, jnp.vdot(bar,w @ k), h*(k @ bar), h*w[:,None]*bar[None,:])
            for actual, reference in zip(bars,expected,strict=True):
                np.testing.assert_allclose(actual,reference,rtol=2e-13,atol=2e-13)
            loss = lambda stages: jnp.sum(endpoint(y,h,w,stages)**2)
            hvp = jax.jit(lambda stages,direction: jax.jvp(jax.grad(loss),(stages,),(direction,))[1])(k,dk)
            np.testing.assert_allclose(hvp,2*h*h*w[:,None]*(w @ dk)[None,:],rtol=2e-13,atol=2e-13)

    def test_reports_endpoint_time_and_compiled_buffers(self):
        jax = self.jax
        def previous(y,h,w,k):
            y,h,w,k = jax.lax.optimization_barrier((y,h,w,k))
            total = jax.lax.optimization_barrier(w @ k)
            return jax.lax.optimization_barrier(y+h*total)
        values = self.arrays()
        for name, function in (("native_reference", previous), ("fixed_tree", self.endpoint)):
            start = time.perf_counter()
            executable = jax.jit(function).lower(*values).compile()
            compile_s = time.perf_counter()-start
            executable(*values).block_until_ready()
            timings = []
            for _ in range(10):
                start = time.perf_counter()
                executable(*values).block_until_ready()
                timings.append(time.perf_counter()-start)
            print(f"[radau-endpoint] JAX={jax.__version__} {name} compile_s={compile_s:.6e} "
                  f"median_s={np.median(timings):.6e} memory={executable.memory_analysis()}",flush=True)


if __name__ == "__main__":
    unittest.main()
