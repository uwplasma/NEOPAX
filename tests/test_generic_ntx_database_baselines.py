"""Generic baseline diagnostic checks using NumPy only; no physical solves."""
from contextlib import redirect_stdout
import csv
import importlib.util
import io
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
