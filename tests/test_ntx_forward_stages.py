"""Observe mocked solver data only; import no JAX or physics packages."""
import dataclasses
from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

SCRIPT=Path(__file__).resolve().parents[1]/'examples/benchmarks/_ntx_forward_stages.py'
spec=importlib.util.spec_from_file_location('forward_stage_capture',SCRIPT)
capture=importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


def fake_solver():
    direct=SimpleNamespace(__file__=__file__)
    factor_calls=[]
    callback_calls=[]
    def factor(a):
        factor_calls.append(a.copy())
        return a.copy(),np.arange(a.shape[0],dtype=np.int32)
    def state(*args,**kwargs):
        matrix=np.asarray([[2.,.1],[.2,3.]])
        direct.lu_factor(matrix)  # A non-head call must not be captured.
        def head_step(j):
            return direct.lu_factor(matrix+j)
        values=[head_step(j) for j in range(3)]
        return np.ones((3,2,2)),np.stack([v[0] for v in values]),np.stack([v[1] for v in values]),np.ones((3,2,2)),np.zeros((3,2,2))
    direct.lu_factor=factor
    direct._block_thomas_truncated_fn_state=state
    def callback(fn,*args,ordered):
        callback_calls.append(ordered)
        fn(*args)
    jax=SimpleNamespace(debug=SimpleNamespace(callback=callback))
    return direct,jax,factor_calls,callback_calls


class ForwardStageCaptureTests(unittest.TestCase):
    def test_observer_keeps_original_state_and_restores_functions(self):
        direct,jax,factors,callbacks=fake_solver()
        original_state,original_factor=direct._block_thomas_truncated_fn_state,direct.lu_factor
        retained,phase={},['baseline']
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with capture.observe_head(root,phase,retained,jax=jax,direct=direct):
                result=direct._block_thomas_truncated_fn_state()
                np.testing.assert_array_equal(retained['head_lu'],result[1])
                np.testing.assert_array_equal(retained['head_sigma'],result[3])
                for j in range(3):
                    with np.load(root/f'baseline_head_{j}.npz',allow_pickle=False) as z:
                        np.testing.assert_array_equal(z['matrix'],np.asarray([[2.,.1],[.2,3.]])+j)
                retained.clear()
                phase[0]='minus'
                direct._block_thomas_truncated_fn_state()
            self.assertEqual(retained,{})
            self.assertEqual(len(list(root.glob('*_head_*.npz'))),6)
        self.assertIs(direct.lu_factor,original_factor)
        self.assertIs(direct._block_thomas_truncated_fn_state,original_state)
        self.assertEqual(len(factors),8)
        self.assertEqual(callbacks,[True]*6)

    def test_exception_restores_original_functions(self):
        direct,jax,_,_=fake_solver()
        original=direct.lu_factor
        retained={}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError,'stop'):
                with capture.observe_head(Path(directory),['baseline'],retained,jax=jax,direct=direct):
                    direct._block_thomas_truncated_fn_state()
                    raise RuntimeError('stop')
        self.assertEqual(retained,{})
        self.assertIs(direct.lu_factor,original)

    def test_capture_saves_same_inputs_and_keeps_observed_primal_drift_visible(self):
        direct,jax,_,_=fake_solver()
        barriers=[]
        jax.jit=lambda f:f
        jax.numpy=np
        jax.block_until_ready=lambda x:x
        jax.effects_barrier=lambda:barriers.append(True)
        Surface=dataclasses.make_dataclass('Surface',[('b_cos',object)])
        template=Surface(np.asarray(2.))
        names=('b_cos',)
        geometry=SimpleNamespace(**{name:np.ones((2,2)) for name in (
            'b','d_b_dtheta','d_b_dzeta','jacobian','b_sub_theta','b_sub_zeta','b_sup_theta','b_sup_zeta',
            'volume_prime','b2_mean','radial_drift_spatial','b0','coefficient_psi_scale','transport_psi_scale')},
            grid=SimpleNamespace(dtheta=.5,dzeta=.7))
        def prepare(surface,grid):
            return SimpleNamespace(surface=surface,geometry=geometry,d_theta=np.eye(2),d_zeta=np.eye(2))
        def solve(prepared,case):
            modes=direct._block_thomas_truncated_fn_state()[0]
            b=prepared.surface.b_cos
            return SimpleNamespace(D11=b*b,D31=b+case.epsi_hat,D13=b-case.epsi_hat,D33=3.*b,
                D33_spitzer=1./case.nu_hat,f1_modes=modes[...,0],f3_modes=modes[...,1])
        ntx=SimpleNamespace(prepare_monoenergetic_system=prepare,solve_prepared=solve,
            MonoenergeticCase=lambda **kw:SimpleNamespace(**kw))
        numbers=tuple((np.asarray(b),np.asarray(.1)) for b in (2.,1.9,2.1))
        ordinary=[np.asarray([b*b,b+.1,b-.1,3*b,1e6]) for b in (2.,1.9,2.1)]
        ordinary[-1]=ordinary[-1].copy();ordinary[-1][2]+=1e-3
        modules={'jax':jax,'jax.numpy':np,'ntx':ntx,'solvax':SimpleNamespace(direct=direct)}
        with tempfile.TemporaryDirectory() as directory,patch.dict(sys.modules,modules),redirect_stdout(io.StringIO()) as output:
            root=Path(directory)
            capture.capture_forward_stages(root,template=template,names=names,grid=None,
                values=numbers[0],minus=numbers[1],plus=numbers[2],nu=np.asarray(1e-6),ordinary=ordinary,fd_step=.1)
            self.assertEqual(len(list(root.glob('*.npz'))),12)
            for i,phase in enumerate(('baseline','minus','plus')):
                with np.load(root/f'{phase}_stages.npz',allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['input.b_cos'],numbers[i][0])
                    np.testing.assert_array_equal(z['ordinary_coefficients'],ordinary[i])
                    self.assertIn('head_sigma',z.files)
                    self.assertIn('f3_modes',z.files)
                    self.assertEqual(float(z['fd_step']),.1)
            self.assertIn('phase=plus coefficients_equal=False',output.getvalue())
        self.assertEqual(barriers,[True]*3)


if __name__=='__main__':
    unittest.main()
