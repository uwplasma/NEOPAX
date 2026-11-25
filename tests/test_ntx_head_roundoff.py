"""Manufactured saved-state error budgets; no physics packages or JAX."""
import importlib.util
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np


path=Path(__file__).resolve().parents[1]/'examples/benchmarks/_ntx_head_roundoff.py'
spec=importlib.util.spec_from_file_location('head_roundoff',path)
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def fixture(component, error):
    diagonal=np.asarray([[3.,5.],[7.,11.],[2.,4.]])
    factors=diagonal.copy()
    if component=='lu_factorization':
        factors[2]+=error*np.asarray([1.,-2.])
    lower=np.asarray([np.zeros((2,2)),[[.2,.1],[.05,.3]],[[.1,.04],[.02,.15]]])
    u0,u1=np.asarray([[.4,.05],[.03,.2]]),np.asarray([[.5,.2],[.04,.3]])
    sigma=np.asarray([[3.,-2.5],[.7,-.2],[1.,.4]])
    x=lower[2]/factors[2,:,None]
    if component=='lu_application':
        x+=error*np.asarray([[1.,2.],[-3.,1.]])
    t=sigma[2]/factors[2]
    d=np.diag(diagonal[1])+u1@x
    b=sigma[1]+u1@t
    if component=='schur_matrix_product':
        d+=error*np.asarray([[1.,2.],[-3.,1.]])
    if component=='schur_source_product':
        b-=error*np.asarray([1.,-2.])
    modes=np.empty((3,2))
    modes[0]=sigma[0]/factors[0]
    modes[1]=(sigma[1]-lower[1]@modes[0])/factors[1]
    modes[2]=(sigma[2]-lower[2]@modes[1])/factors[2]
    state=dict(head_lu=np.stack([np.diag(row) for row in factors]),
        head_pivots=np.tile([0,1],(3,1)),head_lower=lower,
        head_sigma=np.stack((np.zeros_like(sigma),sigma),axis=-1),f3_modes=modes)
    heads=[dict(matrix=np.diag(diagonal[0]),u_j=u0),
           dict(matrix=np.diag(diagonal[1]),d_j=d,u_j=u1,x=x,
                solved_sigma=np.stack((np.zeros_like(t),t),axis=-1),
                b_j=np.stack((np.zeros_like(b),b),axis=-1),
                sigma_next=state['head_sigma'][2]),
           dict(matrix=np.diag(diagonal[2]))]
    return state,heads,np.asarray([[.3,.2],[0.,0.],[-.4,.6]])


class HeadRoundoffTests(unittest.TestCase):
    def test_recorded_transpose_factor_action_respects_pivot_order(self):
        matrix=np.asarray([[1.,3.],[4.,2.]])
        lu=np.asarray([[4.,2.],[.25,2.5]])
        expected=np.asarray([-2.,3.])
        actual=audit.transpose_action(lu,np.asarray([1,1]),matrix.T@expected)
        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-16)

    def test_budget_localizes_independent_injected_arithmetic_errors(self):
        components=('schur_matrix_product','schur_source_product','lu_factorization','lu_application')
        for component in components:
            with self.subTest(component=component):
                result=audit.row_budget(*fixture(component,1e-5))
                self.assertGreater(abs(result[component]),1e-10)
                self.assertLess(abs(result['decomposition_residual']),2e-15)
                self.assertLess(result['adjoint_relative_residual'],2e-5)
                for other in set(components)-{component}:
                    self.assertLess(abs(result[other]),2e-16)

    def test_exact_fixture_has_no_material_arithmetic_error(self):
        result=audit.row_budget(*fixture('none',0.))
        self.assertLess(max(abs(value) for key,value in result.items()
                            if key!='adjoint_relative_residual'),2e-16)

    def test_optional_residual_keeps_the_existing_error_budget(self):
        inputs=fixture('lu_application',1e-5)
        original=audit.row_budget(*inputs)
        result,residual=audit.row_budget(*inputs,return_application_residual=True)
        self.assertEqual(original,result)
        self.assertEqual(residual.shape,(2,))
        self.assertTrue(np.isfinite(residual).all())
        self.assertGreater(float(np.linalg.norm(residual)),1e-8)

    def test_saved_report_keeps_observed_primal_mismatch_visible(self):
        state,heads,_=fixture('none',0.)
        geometry=dict(jacobian=np.ones((1,2)),volume_prime=np.asarray(1.),
                      radial_drift_spatial=np.asarray([[.2,-.3]]),
                      coefficient_psi_scale=np.asarray(1.),b0=np.asarray(1.))
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()) as output:
            root=Path(directory)
            for phase in ('baseline','minus','plus'):
                ordinary=np.asarray([1.,2.,3.,4.,5.])
                actual=ordinary.copy()
                if phase=='plus':
                    actual[2]+=1e-4
                np.savez_compressed(root/f'{phase}_stages.npz',**state,
                    **{'geometry.'+key:value for key,value in geometry.items()},
                    coefficients=actual,ordinary_coefficients=ordinary,
                    grid_dtheta=.2,grid_dzeta=.4,fd_step=1e-3)
                for j,head in enumerate(heads):
                    np.savez_compressed(root/f'{phase}_head_{j}.npz',**head)
            rows=audit.report_head_roundoff(root)
            self.assertEqual([r['observed_equals_ordinary'] for r in rows],[True,True,False])
            self.assertTrue((root/'head_roundoff.csv').is_file())
        self.assertIn('observed_raw_D13_fd=',output.getvalue())
        self.assertIn('observed_matches=[True, True, False]',output.getvalue())


if __name__=='__main__':
    unittest.main()
