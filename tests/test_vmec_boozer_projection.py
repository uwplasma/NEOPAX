"""Test the production Fourier projection without importing NEOPAX or VMEX."""
import ast
import os
from pathlib import Path
import re
import sys
import time
from types import ModuleType
import unittest
from unittest.mock import patch

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]/'NEOPAX/_geometry_autodiff.py'


def production_projection():
    tree = ast.parse(SOURCE.read_text())
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == '_vmec_boozer_fourier_projection')
    builder = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == '_neopax_vmex_boozer_table_inputs_from_state')
    project = next(n for n in builder.body if isinstance(n, ast.FunctionDef) and n.name == 'project')
    namespace = {}
    exec(compile(ast.Module(body=[helper, project], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace


class VmecProjectionNumpyTests(unittest.TestCase):
    def test_weighted_even_odd_projection_keeps_the_equation_and_logical_grid_order(self):
        module = ModuleType('solvax._reproducible')
        shapes = []
        def product(a, b):
            shapes.append((a.shape, b.shape))
            return a @ b
        module.schur_product = product
        rng = np.random.default_rng(83)
        values, cosine, sine = (rng.normal(size=s) for s in ((5,3,7), (3,7,11), (3,7,11)))
        weights = np.full(11, 2./21)
        weights[0] *= .5
        namespace = production_projection()
        with patch.dict(sys.modules, {'solvax._reproducible': module}):
            for dtype in (np.float32, np.float64):
                for reverse in (False, True):
                    a = np.asfortranarray(values.astype(dtype))
                    c, s = cosine.astype(dtype), sine.astype(dtype)
                    if reverse:
                        a, c, s = a[:,::-1,:], c[::-1,:,:], s[::-1,:,:]
                    namespace.update(cos_t=c, sin_t=s, weights_j=weights.astype(dtype))
                    for parity, basis in (('even',c), ('odd',s)):
                        actual = namespace['project'](a, parity)
                        terms = a.astype(np.longdouble)[...,None]*basis.astype(np.longdouble)[None,...]
                        reference = terms.sum(axis=(1,2))*weights.astype(dtype)
                        bound = 64*np.finfo(dtype).eps*np.abs(terms).sum(axis=(1,2))*weights
                        self.assertTrue(np.all(np.abs(actual-reference) <= bound))
                        self.assertEqual(actual.dtype, dtype)
        self.assertEqual(set(shapes), {((5,21),(21,11))})


@unittest.skipUnless(os.environ.get('NEOPAX_TEST_VMEC_PROJECTION_GPU') == '1', 'opt-in CUDA check')
class VmecProjectionCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as jnp
        if jax.default_backend() != 'gpu' or not jax.config.x64_enabled:
            raise RuntimeError('Requires CUDA and JAX_ENABLE_X64=true.')
        cls.jax, cls.jnp = jax, jnp
        cls.project = staticmethod(production_projection()['_vmec_boozer_fourier_projection'])

    def arrays(self):
        if path := os.environ.get('NEOPAX_VMEC_PROJECTION_CAPTURE'):
            with np.load(path, allow_pickle=False) as saved:
                fields = {match[1]: saved[k].copy() for k in saved.files
                          if (match := re.fullmatch(r"trace\['\d+_projection_00_inputs'\]\['(values|basis)'\]", k))}
            if set(fields) != {'values','basis'}:
                raise ValueError('Capture must contain the original projection operands.')
            values, basis = fields['values'], fields['basis']
        else:
            rng = np.random.default_rng(193)
            values, basis = rng.normal(size=(200,18,16)), rng.normal(size=(18,16,128))
        return self.jnp.asarray(values), self.jnp.asarray(basis)

    def test_projection_bits_match_across_jit_layout_and_batching(self):
        jax, jnp, project = self.jax, self.jnp, self.project
        a, b = self.arrays()
        reference = np.asarray(jax.jit(project)(a,b))
        auxiliary = jax.jit(lambda x,y: (project(x,y), jnp.sum(x)))
        transposed = jax.jit(lambda x,y: project(jnp.swapaxes(x,1,2),y))
        batched = jax.jit(jax.vmap(project, in_axes=(0,None)))
        results = [auxiliary(a,b)[0], transposed(jnp.swapaxes(a,1,2),b)]
        results.extend(batched(jnp.stack([a]*3), b))
        for actual in results:
            self.assertEqual(np.asarray(actual).tobytes(), reference.tobytes())
        self.assertIn('solvax_fixed_order_product', str(jax.make_jaxpr(project)(a,b)))

    def test_value_jvp_vjp_and_second_derivatives(self):
        jax, jnp, project = self.jax, self.jnp, self.project
        rng = np.random.default_rng(38)
        a,b,da,db = [rng.normal(size=s) for s in ((4,3,5),(3,5,7),(4,3,5),(3,5,7))]
        args, directions = tuple(map(jnp.asarray,(a,b))), tuple(map(jnp.asarray,(da,db)))
        aa,bb,daa,dbb = a.reshape(4,15),b.reshape(15,7),da.reshape(4,15),db.reshape(15,7)
        y, dy = aa @ bb, daa @ bb + aa @ dbb
        value, dot = jax.jit(lambda p,d: jax.jvp(project,p,d))(args,directions)
        np.testing.assert_allclose(value,y,rtol=3e-13,atol=3e-13)
        np.testing.assert_allclose(dot,dy,rtol=3e-13,atol=3e-13)
        bar = rng.normal(size=(4,7))
        bars = jax.jit(lambda p,c: jax.vjp(project,*p)[1](c))(args,jnp.asarray(bar))
        for actual, expected in zip(bars,((bar @ bb.T).reshape(a.shape),(aa.T @ bar).reshape(b.shape)),strict=True):
            np.testing.assert_allclose(actual,expected,rtol=3e-13,atol=3e-13)
        loss = lambda x,z: .5*jnp.sum(project(x,z)**2)
        hvp = jax.jit(lambda p,d: jax.jvp(jax.grad(loss,argnums=(0,1)),p,d)[1])(args,directions)
        expected = ((dy @ bb.T+y @ dbb.T).reshape(a.shape), (daa.T @ y+aa.T @ dy).reshape(b.shape))
        for actual, other in zip(hvp,expected,strict=True):
            np.testing.assert_allclose(actual,other,rtol=3e-12,atol=3e-12)

    def test_reports_time_and_compiled_buffers(self):
        jax, jnp = self.jax, self.jnp
        args = self.arrays()
        for name, function in (('native_reference',lambda a,b: jnp.einsum('rtz,tzm->rm',a,b)),
                               ('fixed_product',self.project)):
            executable = jax.jit(function).lower(*args).compile()
            executable(*args).block_until_ready()
            samples = []
            for _ in range(25):
                start = time.perf_counter()
                executable(*args).block_until_ready()
                samples.append(time.perf_counter()-start)
            print(f'[vmec-projection] JAX={jax.__version__} {name} median_s={np.median(samples):.6e} '
                  f'memory={executable.memory_analysis()}',flush=True)


if __name__ == '__main__':
    unittest.main()
