"""Analytic saved-field diagnostics; import no JAX or physics packages."""
from contextlib import redirect_stdout
import dataclasses
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

SCRIPT=Path(__file__).resolve().parents[1]/'examples/benchmarks/_ntx_prepared_field_check.py'
spec=importlib.util.spec_from_file_location('prepared_field_check',SCRIPT)
check=importlib.util.module_from_spec(spec);spec.loader.exec_module(check)


class PreparedFieldCheckTests(unittest.TestCase):
    def test_contract_rows_uses_all_fields_and_case_coordinate(self):
        bars={'b':np.asarray([[1.,2.],[-3.,4.]]),'b0':np.asarray([2.,-1.])}
        actual=check.contract_field_rows(bars,{'b':np.asarray([.2,.3]),'b0':np.asarray(.4)},np.asarray([.5,-.2]),.7)
        np.testing.assert_allclose(actual,[1.95,.06],atol=1e-15)
        with self.assertRaisesRegex(ValueError,'prepared field'):
            check.contract_field_rows(bars,{'b':np.asarray([.2,.3])},np.asarray([.5,-.2]),.7)
        with self.assertRaises(FloatingPointError):
            check.contract_field_rows(bars,{'b':np.asarray([np.nan,.3]),'b0':np.asarray(.4)},np.asarray([.5,-.2]),.7)

    def test_saved_field_secants_localize_an_injected_preparation_error(self):
        Surface=dataclasses.make_dataclass('Surface',[('b_cos',object)])
        Geometry=dataclasses.make_dataclass('Geometry',[('b',object),('b0',object)])
        Prepared=dataclasses.make_dataclass('Prepared',[('geometry',object),('d_theta',object),('d_zeta',object)])
        def prepare(surface,grid):
            return Prepared(Geometry(surface.b_cos**2,np.sum(surface.b_cos)),np.eye(2),np.eye(2))
        def coefficients(g,eps):
            total=np.sum(g.b)
            return np.asarray([total+2*g.b0+.5*eps,0.,3*total-g.b0+2*eps,-total+4*g.b0-eps,0.])
        def pullback(prepared,case,basis):
            return SimpleNamespace(epsi_hat=np.asarray([.5,2.,-1.])),Geometry(
                np.asarray([[1.,1.],[3.,3.],[-1.,-1.]]),np.asarray([2.,-1.,4.]))
        def jvp(f,p,t):
            x,d=p[0],t[0]
            return f(*p),(2*x[0]*d[0],np.sum(d[0]))
        jax=SimpleNamespace(numpy=np,jit=lambda f:f,block_until_ready=lambda x:x,jvp=jvp)
        ntx=SimpleNamespace(prepare_monoenergetic_system=prepare,MonoenergeticCase=lambda **kw:SimpleNamespace(**kw),
            _neopax_ad=SimpleNamespace(pullback_prepared_database_coefficients_to_vmec_primitive_multi_rhs=pullback))
        solver=SimpleNamespace(_prepared_implicit_vjp_primal=lambda p,nu,eps:(coefficients(p.geometry,eps),))
        modules={'jax':jax,'jax.numpy':np,'ntx':ntx,'ntx._solver':solver}
        x=np.asarray([1.,2.]);dx=np.asarray([.2,.3]);eps=.2;de=.4;h=1e-4
        true_geo=Geometry(x*x,np.sum(x))
        raw_baseline=coefficients(true_geo,eps)[[0,2,3]]
        fd_b=2*x*dx+np.asarray([.01,-.02]);fd_b0=np.sum(dx)+.03
        with tempfile.TemporaryDirectory() as directory,patch.dict(sys.modules,modules),redirect_stdout(io.StringIO()):
            root=Path(directory);reference=root/'reference';reference.mkdir()
            for phase,sign in (('baseline',0),('minus',-1),('plus',1)):
                geometry=Geometry(true_geo.b+sign*h*fd_b,true_geo.b0+sign*h*fd_b0)
                raw=coefficients(geometry,eps+sign*h*de)
                np.savez(reference/f'{phase}_stages.npz',**{'input.b_cos':x+sign*h*dx,
                    'input.epsi_hat':np.asarray(eps+sign*h*de),'geometry.b':geometry.b,'geometry.b0':geometry.b0,
                    'coefficients':raw,'ordinary_coefficients':raw,'fd_step':h,'nu':1e-6,
                    'd_theta':np.eye(2),'d_zeta':np.eye(2)})
            rows=check.check_prepared_fields(root/'out',reference,template=Surface(x),names=('b_cos',),grid=None,
                values=(x,np.asarray(eps)),direction=(dx,np.asarray(de)),nu=np.asarray(1e-6),h=h,raw_jvp=np.asarray([2.8,5.1,0.]),raw_baseline=raw_baseline)
            np.testing.assert_allclose([r['production_vjp'] for r in rows],[2.8,5.1,0.],atol=2e-15)
            np.testing.assert_allclose([r['field_fd_minus_jvp'] for r in rows],[.05,-.06,.13],atol=2e-11)
            self.assertLess(max(abs(r['saved_fd_minus_field_fd']) for r in rows),3e-11)
            self.assertTrue((root/'out/prepared_field_bars.npz').is_file())
            with self.assertRaisesRegex(ValueError,'coefficient baseline differs'):
                check.check_prepared_fields(root/'bad_baseline',reference,template=Surface(x),names=('b_cos',),grid=None,
                    values=(x,np.asarray(eps)),direction=(dx,np.asarray(de)),nu=np.asarray(1e-6),h=h,
                    raw_jvp=np.zeros(3),raw_baseline=raw_baseline+1.)
            for bad_h,bad_values in ((h*2,(x,np.asarray(eps))),(h,(x+.1,np.asarray(eps)))):
                with self.assertRaises(ValueError):
                    check.check_prepared_fields(root/'bad',reference,template=Surface(x),names=('b_cos',),grid=None,
                        values=bad_values,direction=(dx,np.asarray(de)),nu=np.asarray(1e-6),h=bad_h,raw_jvp=np.zeros(3),raw_baseline=raw_baseline)
            (reference/'metadata.json').write_text('{"sources":{"ntx._solver":{"sha256":"different"}}}')
            with self.assertRaisesRegex(ValueError,'source differs'):
                check.check_prepared_fields(root/'bad_source',reference,template=Surface(x),names=('b_cos',),grid=None,
                    values=(x,np.asarray(eps)),direction=(dx,np.asarray(de)),nu=np.asarray(1e-6),h=h,raw_jvp=np.zeros(3),raw_baseline=raw_baseline)


if __name__=='__main__':unittest.main()
