"""NumPy checks of column batching only; import no JAX or physics packages."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np


folder = Path(__file__).resolve().parents[1]/'examples/benchmarks'
sys.path.insert(0,str(folder))
spec = importlib.util.spec_from_file_location('saved_lu_layout',folder/'check_saved_lu_rhs_layout.py')
audit = importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


class SavedLuLayoutTests(unittest.TestCase):
    def test_native_comparison_checks_signed_zero_and_dtype(self):
        a=np.asarray([0.,1.])
        self.assertTrue(audit.same_bytes(a,a.copy()))
        self.assertFalse(audit.same_bytes(a,np.asarray([-0.,1.])))
        self.assertFalse(audit.same_bytes(a,a.astype(np.float32)))

    def test_panels_keep_all_columns_and_reuse_identical_factors(self):
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
        self.assertNotIn('jax',sys.modules)

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
