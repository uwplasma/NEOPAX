"""NumPy checks of column batching only; import no JAX or physics packages."""
import importlib.util
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


folder = Path(__file__).resolve().parents[1]/'examples/benchmarks'
sys.path.insert(0,str(folder))
spec = importlib.util.spec_from_file_location('saved_lu_layout',folder/'check_saved_lu_rhs_layout.py')
audit = importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


class SavedLuLayoutTests(unittest.TestCase):
    def test_ad_timing_includes_primal_and_checks_analytic_duality(self):
        jax_was_loaded='jax' in sys.modules
        stats=SimpleNamespace(temp_size_in_bytes=123,argument_size_in_bytes=456,output_size_in_bytes=789)
        class Executable:
            def __init__(self,fn): self.fn=fn
            def __call__(self,*args): return self.fn(*args)
            def as_text(self): return 'mock compiler IR'
            def memory_analysis(self): return stats
        class Jitted:
            def __init__(self,fn): self.fn=fn
            def lower(self,*args): return SimpleNamespace(compile=lambda:Executable(self.fn))
        def jvp(fn,primals,dots):
            a,b,c=primals;da,db,dc=dots
            return fn(a,b,c),(da@b+a@db,da@c+a@dc)
        def vjp(fn,a,b,c):
            def pull(bars):
                left,right=bars
                return left@b.T+right@c.T,a.T@left,a.T@right
            return fn(a,b,c),pull
        jax=SimpleNamespace(jit=Jitted,jvp=jvp,vjp=vjp,block_until_ready=lambda x:x)
        a=np.asarray([[2.,.2],[.1,3.]])
        b=np.asarray([[1.,2.],[3.,4.]])
        c=np.asarray([[.4],[.5]])
        function=lambda lu,piv,matrix,source:(lu@matrix,lu@source)
        reference=function(a,None,b,c)
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()) as output:
            rows=audit.measure_ad(function,(a,None,b,c),reference,Path(directory),'toy',jax=jax,jnp=np)
            self.assertEqual([r['mode'] for r in rows],['primal_and_jvp','primal_and_vjp'])
            self.assertEqual([r['temp_bytes'] for r in rows],[123,123])
            self.assertEqual(len(list(Path(directory).glob('*_compiled.txt'))),2)
        self.assertIn('primal_matches=[True, True]',output.getvalue())
        error=float(output.getvalue().split('JVP_VJP_duality_relative_error=')[-1])
        self.assertLess(error,1e-14)
        self.assertEqual('jax' in sys.modules,jax_was_loaded)

    def test_native_comparison_checks_signed_zero_and_dtype(self):
        a=np.asarray([0.,1.])
        self.assertTrue(audit.same_bytes(a,a.copy()))
        self.assertFalse(audit.same_bytes(a,np.asarray([-0.,1.])))
        self.assertFalse(audit.same_bytes(a,a.astype(np.float32)))

    def test_panels_keep_all_columns_and_reuse_identical_factors(self):
        jax_was_loaded='jax' in sys.modules
        factors = object()
        lax = SimpleNamespace(map=lambda fn,blocks:np.stack([fn(b) for b in blocks]))
        for count in (1,2,7,21):
            rhs = np.arange(5*count,dtype=float).reshape(5,count)+.25
            for panel in (0,1,2,8,16,32):
                calls = []
                def solve(actual,b):
                    self.assertIs(actual,factors)
                    calls.append(b.shape[1])
                    return b*np.arange(1.,6.)[:,None]
                actual = audit.panel_solve(solve,factors,rhs,panel,xp=np,lax=lax)
                np.testing.assert_array_equal(actual,rhs*np.arange(1.,6.)[:,None])
                self.assertEqual(sum(calls),count)
                if panel:
                    self.assertLessEqual(max(calls),panel)
        self.assertEqual('jax' in sys.modules,jax_was_loaded)

    def test_empty_and_invalid_inputs_do_not_call_a_solve(self):
        def forbidden(*args):
            self.fail('Unexpected solver call')
        result = audit.panel_solve(forbidden,None,np.zeros((5,0)),1,xp=np,lax=None)
        self.assertEqual(result.shape,(5,0))
        for rhs,panel in ((np.ones(5),1),(np.ones((5,2)),-1)):
            with self.assertRaises(ValueError):
                audit.panel_solve(forbidden,None,rhs,panel,xp=np,lax=None)


if __name__ == '__main__':
    unittest.main()
