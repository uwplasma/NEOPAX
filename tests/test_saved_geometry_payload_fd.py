"""Geometry-input secants with actual cost bars; NumPy only, no JAX import."""
import dataclasses
import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1]/'examples/benchmarks/check_saved_geometry_payload_fd.py'
sys.path.insert(0,str(SCRIPT.parent))
loaded_jax = 'jax' in sys.modules
spec = importlib.util.spec_from_file_location('geometry_payload_fd',SCRIPT)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class GeometryMapTests(unittest.TestCase):
    def test_all_input_groups_and_cost_rows_match_independent_nonlinear_secants(self):
        Geometry = dataclasses.make_dataclass('Geometry',[('v',object)])
        x,dx,h0 = .7,.4,.01
        paths = ("['geometry'].v","['channels'].c","['surfaces'][0].b_cos")
        weights = np.asarray([[2.,-1.,.3],[-.4,3.,2.]])
        derivatives = np.asarray([3*x*x*dx,2*x*dx,np.cos(x)*dx])
        expected_ad = weights@derivatives
        direction = {k:np.asarray(v) for k,v in zip(paths,derivatives)}
        capture = {'parameter_labels[0]':np.asarray('RBC:1:0'),
            'parameter_labels[1]':np.asarray('ZBS:1:0'),
            'geometry_gradients':np.column_stack((np.asarray([100.,200.]),expected_ad))}
        for i in range(2):
            capture.update({f'payload_bars[{i}]'+k:np.asarray(v) for k,v in zip(paths,weights[i])})
        def inputs(step):
            z = x+dx*step
            return dict(geometry=Geometry(np.asarray(z**3)),channels={'c':np.asarray(z*z)},
                        surfaces=(SimpleNamespace(b_cos=np.asarray(np.sin(z))),))
        refs = {name:{'v':inputs(step)['geometry'].v} for name,step in
                (('baseline',0.),('minus',-h0),('plus',h0))}
        report = dict(objective_labels=['cost_a','cost_b'],fd_step=h0,gradient_fd=[2.,3.])
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            args = dict(capture=capture,direction=direction,parameter='ZBS:1:0',report=report,
                        references=refs,out_dir=Path(directory))
            rows = check.run_sweep(inputs,**args)
            self.assertEqual(len(rows),8)
            for row in rows:
                h = row['fd_step']
                numerical = np.asarray([3*x*x*dx+h*h*dx**3,2*x*dx,np.cos(x)*np.sin(h*dx)/h])
                i = report['objective_labels'].index(row['objective'])
                self.assertAlmostEqual(row['geometry_map_fd'],float(weights[i]@numerical),places=12)
                self.assertAlmostEqual(sum(row[g+'_fd'] for g in ('geometry','channels','surfaces')),row['geometry_map_fd'],places=12)
                self.assertEqual(row['saved_ad'],expected_ad[i])
                self.assertEqual(row['map_minus_full_fd_at_same_step'] is None,row['step_scale']!=1.)
            self.assertTrue((Path(directory)/'geometry_map_fields.csv').is_file())
            with np.load(Path(directory)/'payload_minus_h1.npz') as archive:
                np.testing.assert_array_equal(archive[paths[0]],inputs(-h0)['geometry'].v)
            # Bad saved derivatives must fail before calling the constructor.
            wrong = dict(direction);wrong[paths[0]] = wrong[paths[0]]+.1
            with self.assertRaisesRegex(ValueError,'reconstruct'):
                check.run_sweep(lambda _:self.fail('Constructor reached'),**dict(args,direction=wrong))
            # A different replayed h0 endpoint cannot be used as an explanation
            # of the previously recorded full FD result.
            bad_refs = dict(refs,plus={'v':refs['plus']['v']+1e-10})
            with self.assertRaisesRegex(ValueError,'Recorded plus geometry'):
                check.run_sweep(inputs,**dict(args,references=bad_refs))

    def test_path_resolution_rejects_unknown_fields_and_nonfinite_or_integer_values(self):
        item = dict(geometry=SimpleNamespace(a=np.asarray(2.)),channels={'b':np.asarray(3.)},
                    surfaces=(SimpleNamespace(b_cos=np.asarray([4.])),))
        paths = ("['geometry'].a","['channels'].b","['surfaces'][0].b_cos")
        arrays = check.payload_arrays(item,paths)
        self.assertEqual(tuple(arrays),paths)
        for path in ("['geometry'].a.extra","__import__('os')","['other'].a"):
            with self.assertRaisesRegex(ValueError,'Unsupported'):
                check.payload_arrays(item,[path])
        for bad in (np.asarray(np.nan),np.asarray(1)):
            with self.assertRaisesRegex(ValueError,'Nonfinite or non-floating'):
                check.payload_arrays(dict(item,channels={'b':bad}),[paths[1]])
        capture = {'geometry_gradients':np.ones((1,1)),"payload_bars[0]['geometry'].a":np.asarray([1.,2.])}
        with self.assertRaisesRegex(ValueError,'Invalid'):
            check.project(capture,{paths[0]:np.asarray(2.)})
        self.assertEqual('jax' in sys.modules,loaded_jax)

    def test_cli_rejects_different_recorded_spectral_state_before_jax_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ad,fd = root/'ad/snapshots',root/'fd_ZBS_1_0/snapshots'
            ad.mkdir(parents=True);fd.mkdir(parents=True)
            terminal = {k:np.asarray([1.]) for k in ('initial_y','final_y','final_time','values')}
            terminal.update({'labels[0]':np.asarray('cost'),"support['geometry'].v":np.asarray(2.)})
            np.savez(ad/'ad_terminal_one.npz',**terminal)
            np.savez(fd/'fd_baseline_one.npz',**terminal)
            np.savez(ad/'ad_compact_geometry_one.npz',**{'state.R_cos':np.asarray([1.])})
            np.savez(ad/'ad_compact_tangents_one.npz',**{'parameter_labels[0]':np.asarray('ZBS:1:0')})
            np.savez(fd/'fd_geometry_linearization_one.npz',parameter_label=np.asarray('ZBS:1:0'),
                     **{'state.R_cos':np.asarray([2.])})
            (fd.parent/'summary.json').write_text(json.dumps(dict(parameter_name='ZBS:1:0',
                geometry_fd_lane='frozen_linearized',objective_labels=['cost'])))
            with patch.object(sys,'argv',[str(SCRIPT),'--run-dir',str(root),'--out-dir',str(root/'result')]), \
                 self.assertRaisesRegex(ValueError,'Recorded state.'):
                check.main()
        self.assertEqual('jax' in sys.modules,loaded_jax)


if __name__ == '__main__':
    unittest.main()
