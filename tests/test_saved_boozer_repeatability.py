"""Check saved argument fidelity and replay isolation without importing JAX."""
from contextlib import contextmanager, redirect_stdout
from dataclasses import make_dataclass
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

FILE = Path(__file__).resolve().parents[1]/'examples/benchmarks/check_saved_boozer_repeatability.py'
with patch.object(sys,'path',[str(FILE.parent),*sys.path]):
    spec = importlib.util.spec_from_file_location('boozer_replay',FILE)
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)


def capture():
    return {'capture.inputs.bmnc':np.asarray([[1.,-0.]],dtype=np.float64),
            'capture.constants.nfp':np.asarray(2), 'capture.constants.asym':np.asarray(False),
            'capture.grids.theta_grid':np.asarray([0.,1.]),
            'capture.grids.theta_grid.__weak_type__':np.asarray(True),
            'capture.surface_indices':np.asarray([0],dtype=np.int32),
            'capture.jit':np.asarray(True), 'capture.outputs.bmnc_b':np.asarray([1.])}


def api_types():
    return SimpleNamespace(BoozXformConstants=make_dataclass('Constants',['nfp','asym']),
                           BoozXformGrids=make_dataclass('Grids',['theta_grid']))


class BoozerReplayTests(unittest.TestCase):
    def test_reconstruction_preserves_bytes_dtypes_weak_grids_and_static_constants(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'capture.npz'
            original=capture()
            np.savez(path,**original)
            data,groups=replay.read_capture(path)
        conversions=[]
        def array(value,*,weak_type):
            conversions.append((value,weak_type))
            return value.copy()
        args=replay.restore_arguments(data,groups,api_types(),array)
        self.assertEqual(args['inputs'].bmnc.tobytes(),original['capture.inputs.bmnc'].tobytes())
        self.assertEqual(args['surface_indices'].dtype,np.int32)
        self.assertIs(type(args['constants'].nfp),int)
        self.assertIs(args['constants'].asym,False)
        self.assertIs(args['jit'],True)
        self.assertEqual([weak for _,weak in conversions],[True,False,False])
        self.assertEqual(set(groups['grids']),{'theta_grid'})

    def test_invalid_captures_fail_before_a_worker_or_jax_import(self):
        for key,value in (('capture.jit',False),('capture.inputs.bmnc',np.asarray([np.nan]))):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as directory:
                data=capture();data[key]=np.asarray(value)
                path=Path(directory)/'capture.npz';np.savez(path,**data)
                with self.assertRaises(ValueError):
                    replay.read_capture(path)
        self.assertNotIn('jax',sys.modules)

    def test_worker_reuses_original_api_arguments_and_retains_first_changed_output(self):
        class Array(np.ndarray):
            weak_type=False
        def convert(value,dtype,*,weak_type):
            result=np.asarray(value,dtype=dtype).view(Array)
            result.weak_type=weak_type
            return result
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'capture.npz';np.savez(path,**capture())
            data,groups=replay.read_capture(path)
            api=api_types();api.__file__=str(path)
            (root/'loaded_sources.json').write_text(json.dumps({'booz_xform_jax.jax_api':
                {'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}))
            calls=[]
            def original(**kwargs):
                calls.append(kwargs)
                return {'bmnc_b':np.asarray([1. if len(calls)<3 else np.nextafter(1.,2.)]),'unused':None}
            api.booz_xform_from_inputs=original
            @contextmanager
            def compiler_capture(folder,compiler):
                folder.mkdir(parents=True)
                yield
                (folder/'programs.json').write_text(json.dumps([{'name':'jit_booz_xform_jax_impl'}]))
            modules={'jax':SimpleNamespace(default_backend=lambda:'gpu',config=SimpleNamespace(x64_enabled=True),
                       devices=lambda:['cuda:0'],__version__='test',block_until_ready=lambda x:x),
                     'jax.numpy':SimpleNamespace(asarray=np.asarray),
                     'jax._src':SimpleNamespace(compiler=object()),
                     'jax._src.lax.lax':SimpleNamespace(_convert_element_type=convert),
                     'booz_xform_jax':SimpleNamespace(jax_api=api),'booz_xform_jax.jax_api':api,
                     'check_saved_ntx_point_repeatability':SimpleNamespace(capture_ntx_ir=compiler_capture)}
            modules['jax'].numpy=modules['jax.numpy']
            args=SimpleNamespace(capture=path,out_dir=root/'out',worker='process_1',repeats=3)
            with patch.dict(sys.modules,modules),redirect_stdout(io.StringIO()):
                replay.worker(args,data,groups)
            self.assertEqual(len(calls),3)
            for kwargs in calls[1:]:
                for key in calls[0]:
                    self.assertIs(kwargs[key],calls[0][key])
            rows=json.loads((args.out_dir/'comparison.json').read_text())
            self.assertEqual([bool(row['vs_first']) for row in rows],[False,False,True])
            with np.load(args.out_dir/'first_changed_output.npz') as saved:
                self.assertEqual(saved['bmnc_b'][0],np.nextafter(1.,2.))
            self.assertTrue(calls[0]['grids'].theta_grid.weak_type)
        self.assertNotIn('jax',sys.modules)

    def test_parent_runs_sequential_fresh_processes_and_reports_cross_process_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'capture.npz';np.savez(path,**capture())
            commands=[]
            class Process:
                def __init__(self,command,**kwargs):
                    commands.append(command)
                    folder=Path(command[command.index('--out-dir')+1])
                    np.savez(folder/'first_output.npz',bmnc_b=np.asarray([1. if len(commands)==1 else np.nextafter(1.,2.)]))
                    self.stdout=['worker complete\n']
                def __enter__(self): return self
                def __exit__(self,*args): return False
                def wait(self): return 0
            out=root/'out'
            with patch.object(sys,'argv',[str(FILE),'--capture',str(path),'--out-dir',str(out),'--repeats','2']), \
                 patch.object(replay.subprocess,'Popen',Process),redirect_stdout(io.StringIO()):
                replay.main()
            self.assertEqual([c[-1] for c in commands],['process_1','process_2'])
            self.assertTrue(all(c[1]==str(FILE) for c in commands))
            rows=json.loads((out/'process_comparison.json').read_text())
            self.assertEqual(rows[0]['field'],'bmnc_b')
            self.assertEqual(rows[0]['different_entries'],1)
        self.assertNotIn('jax',sys.modules)


if __name__=='__main__':
    unittest.main()
