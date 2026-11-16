"""Generic baseline diagnostic checks using NumPy only; no physical solves."""
from contextlib import redirect_stdout
from dataclasses import dataclass
import csv
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("generic_baselines",
    ROOT/"examples/benchmarks/check_generic_ntx_database_baselines.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class GenericDatabaseBaselineTests(unittest.TestCase):
    def test_input_trace_observes_real_boundaries_and_restores_hooks(self):
        @dataclass
        class State:
            R_cos: np.ndarray
        @dataclass
        class Channels:
            a_b: np.ndarray
        state = State(np.array([1.]))
        frozen = State(np.array([2.]))
        calls = []
        norms = SimpleNamespace(volume=np.array(3.), vp=np.array([4.]))
        module = SimpleNamespace(vmec_force_norms_from_bcovar_dynamic=lambda **kw: norms)
        api = SimpleNamespace(booz_xform_from_inputs=lambda **kw: {'rmnc_b': kw['inputs'].R_cos})
        def solve(*args, **kwargs):
            calls.append('solve')
            return state
        def geometry(context, value, **kwargs):
            self.assertIs(geo._import_vmec_module('residue').vmec_force_norms_from_bcovar_dynamic(), norms)
            result = api.booz_xform_from_inputs(inputs=value, constants=SimpleNamespace(nfp=2), jit=True)
            return SimpleNamespace(R0=result['rmnc_b'])
        def scan(context, value, g, **kwargs):
            api.booz_xform_from_inputs(inputs=value, constants=SimpleNamespace(nfp=2), jit=True)
            return Channels(g.R0), (value,)
        geo = SimpleNamespace(_solve_state_for_single_param=solve,
            _build_neopax_geometry_from_state=geometry, _import_vmec_module=lambda _: module,
            build_ntx_runtime_scan_inputs_from_vmec_state=scan)
        def runtime(config):
            value = geo._solve_state_for_single_param('context', 0., lane='ad')
            g = geo._build_neopax_geometry_from_state('context', value, n_r=51)
            geo.build_ntx_runtime_scan_inputs_from_vmec_state('context', value, g, rho_scan=[.875])
            self.fail('NTX coefficient construction must not be reached')
        for seed in (None, frozen):
            captured = check.trace_inputs({}, geo, api, runtime, lambda c: {'Es': c.a_b*3}, seed)
            self.assertEqual(list(captured), ['00_equilibrium', '01_geometry_equilibrium', '02_volume_inputs',
                '03_boozer_0_inputs', '04_boozer_0_outputs', '05_geometry', '06_boozer_1_inputs',
                '07_boozer_1_outputs', '08_channels', '09_surfaces', '10_electric_fields'])
            expected = state if seed is None else seed
            self.assertIs(captured['00_equilibrium']['R_cos'], expected.R_cos)
            self.assertIs(captured['02_volume_inputs']['volume'], norms.volume)
            self.assertTrue(captured['03_boozer_0_inputs']['jit'])
            np.testing.assert_array_equal(captured['10_electric_fields']['Es'], expected.R_cos*3)
            self.assertIs(geo._solve_state_for_single_param, solve)
            self.assertIs(geo._build_neopax_geometry_from_state, geometry)
            self.assertIs(geo.build_ntx_runtime_scan_inputs_from_vmec_state, scan)
        self.assertEqual(calls, ['solve'])
        with self.assertRaisesRegex(RuntimeError, 'pre-NTX boundary'):
            check.trace_inputs({}, geo, api, lambda _: None, lambda _: {})
        self.assertIs(geo._solve_state_for_single_param, solve)

    def test_trace_reports_first_changed_boundary_and_exact_float_bits(self):
        reference = {"trace['00_equilibrium']['R_cos']": np.array([1.]),
            "trace['04_boozer_0_outputs']['rmnc_b']": np.array([2.]),
            "trace['10_electric_fields']['Es']": np.array([3.])}
        changed = {k: v.copy() for k,v in reference.items()}
        changed["trace['04_boozer_0_outputs']['rmnc_b']"][0] = np.nextafter(2., np.inf)
        with redirect_stdout(io.StringIO()):
            rows = check.compare(reference, changed, 'repeat')
        self.assertEqual(check.first_changed_stage(rows), '04_boozer_0_outputs')
        row = next(r for r in rows if not r['arrays_match'])
        self.assertEqual(row['reference_hex'], float(2.).hex())
        self.assertEqual(row['actual_hex'], float(np.nextafter(2., np.inf)).hex())
        self.assertEqual(check.first_changed_stage(check.compare(reference, reference, 'same')), 'none')
        changed["trace['11_extra']['value']"] = np.ones(1)
        with redirect_stdout(io.StringIO()):
            rows = check.compare(reference, changed, 'extra')
        self.assertEqual(len(rows), 4)
        self.assertTrue(any(not r['shape_matches'] for r in rows))

    def test_reused_state_loader_accepts_both_saved_formats_and_rejects_invalid_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'state.npz'
            for prefix in ('', 'implicit__'):
                np.savez(path, **{prefix+'R_cos': np.array([[1.,2.]])})
                np.testing.assert_array_equal(check.load_named_state(path, ('R_cos',))['R_cos'], [[1.,2.]])
            with self.assertRaisesRegex(ValueError, 'required implicit fields'):
                check.load_named_state(path, ('Z_sin',))
            np.savez(path, implicit__R_cos=np.array([np.inf]))
            with self.assertRaises(FloatingPointError):
                check.load_named_state(path, ('R_cos',))

    def test_trace_parent_repeats_identical_configuration_and_forwards_frozen_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'trace'
            state = Path(temporary)/'state.npz'
            np.savez(state, implicit__R_cos=np.ones(1))
            def complete(command, **kwargs):
                folder = Path(command[command.index('--out-dir')+1])
                self.assertEqual(command[command.index('--worker')+1], 'fd')
                self.assertIn('--trace-inputs', command)
                self.assertEqual(command[command.index('--reuse-states')+1], str(state))
                np.savez(folder/'normalization_inputs_saved.npz', **{"trace['00_equilibrium']['R_cos']": np.ones(1)})
                (folder/'metadata.json').write_text(json.dumps(dict(config_sha256='config', input_sha256='input',
                    reused_state_sha256='state', versions={}, sources={})))
                return SimpleNamespace(returncode=0)
            with patch.object(sys, 'argv', [str(check.__file__), '--out-dir', str(output),
                    '--trace-inputs', '--reuse-states', str(state)]), \
                 patch.object(check.subprocess, 'run', side_effect=complete) as run, redirect_stdout(io.StringIO()) as log:
                check.main()
            self.assertEqual(run.call_count, 3)
            self.assertTrue((output/'baseline_3').is_dir())
            self.assertIn('first_changed_stage=none', log.getvalue())
            self.assertNotIn('jax', sys.modules)

    def test_comparison_separates_tables_inputs_and_nonfinite_masks(self):
        reference = {"support['geometry'].G_PS": np.array([np.nan, 1.]),
            "support['database'].D13": np.array([1.95343806242373]),
            "scan_inputs['channels'].rho": np.array([.75, .875]),
            "scan_inputs['source']": np.asarray("same-input")}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"snapshot.npz"
            np.savez(path, **reference, **{"explicit_support_bars[0]['database'].D13": np.ones(1)})
            loaded = check.saved_arrays(path)
        self.assertEqual(set(loaded), set(reference))
        with redirect_stdout(io.StringIO()):
            rows = check.compare(reference, loaded, "same")
        self.assertTrue(all(r["arrays_match"] and r["max_abs_difference"] == 0 for r in rows))
        changed = {k: v.copy() for k, v in reference.items()}
        changed["support['database'].D13"][0] = 1.9559692059134934
        changed["support['geometry'].G_PS"][0] = 0.
        del changed["scan_inputs['channels'].rho"]
        with redirect_stdout(io.StringIO()):
            rows = {r["leaf"]: r for r in check.compare(reference, changed, "changed")}
        self.assertAlmostEqual(rows["support['database'].D13"]["max_abs_difference"],
                               .0025311434897634655, places=15)
        self.assertTrue(np.isinf(rows["support['geometry'].G_PS"]["max_abs_difference"]))
        self.assertFalse(rows["support['geometry'].G_PS"]["nonfinite_values_match"])
        self.assertFalse(rows["scan_inputs['channels'].rho"]["shape_matches"])

    def test_generic_routes_change_only_scan_recording_and_skip_initialization(self):
        original = {"general": {"mode": "transport"}, "geometry": {"vmec_lane": "ad"},
            "neoclassical": {"flux_model": "ntx_scan_runtime"}}
        ad = check.baseline_config(original, "ad")
        fd = check.baseline_config(original, "fd")
        self.assertEqual(original["general"]["mode"], "transport")
        self.assertEqual(ad["geometry"], fd["geometry"])
        self.assertEqual(ad["general"]["mode"], "ambipolarity")
        self.assertEqual(fd["general"]["mode"], "ambipolarity")
        self.assertEqual(ad["neoclassical"]["ntx_scan_coefficient_reverse_mode"], "structured")
        self.assertTrue(ad["neoclassical"]["ntx_scan_record_primal"])
        self.assertEqual(fd["neoclassical"], original["neoclassical"])
        original["neoclassical"]["interpolation_mode"] = "preprocessed_3d_radial_ntss1d"
        with self.assertRaisesRegex(ValueError, "GENERIC"):
            check.baseline_config(original, "fd")

    def test_worker_refuses_cpu_before_importing_physics(self):
        with patch.dict(sys.modules, jax=SimpleNamespace(default_backend=lambda: "cpu")):
            with self.assertRaisesRegex(RuntimeError, "no CPU fallback"):
                check.worker(SimpleNamespace())

    def test_parent_runs_three_separate_workers_and_compares_their_saved_tables(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/"baselines"

            def complete(command, **kwargs):
                folder = Path(command[command.index("--out-dir")+1])
                offset = {"ad": 0., "fd_1": .01, "fd_2": .02}[folder.name]
                arrays = {key: np.array([1.+offset]) for key in check.TABLES}
                np.savez(folder/"generic_database_baseline_saved.npz", **arrays)
                return SimpleNamespace(returncode=0)

            with patch.object(sys, "argv", [str(check.__file__), "--out-dir", str(output)]), \
                 patch.object(check.subprocess, "run", side_effect=complete) as run, \
                 redirect_stdout(io.StringIO()):
                check.main()
            self.assertEqual(run.call_count, 3)
            routes = [call.args[0][-1] for call in run.call_args_list]
            self.assertEqual(routes, ["ad", "fd", "fd"])
            with (output/"comparison.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 9)
            self.assertEqual({r["comparison"] for r in rows},
                {"fresh_ad_vs_fd_1", "fresh_ad_vs_fd_2", "fd_1_vs_fd_2"})
            self.assertTrue(all("e" in r["max_abs_difference"] for r in rows))
            self.assertNotIn("jax", sys.modules)


if __name__ == "__main__":
    unittest.main()
