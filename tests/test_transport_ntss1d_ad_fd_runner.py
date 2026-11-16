"""Saved-array comparison and dry-run checks; no JAX or physical solvers."""
from contextlib import redirect_stdout
import csv
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT/"examples/benchmarks/benchmark_transport_ntss1d_ad_fd.py"
spec = importlib.util.spec_from_file_location("ntss1d_runner", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class NTSS1DComparisonRunnerTests(unittest.TestCase):
    def saved_case(self, root):
        ad = root/"ad"
        (ad/"snapshots").mkdir(parents=True)
        labels = ["softmax_Er", "bootstrap_current_softmax_abs_scaled"]
        columns = ["vmec:"+p if ":" in p else p for p in runner.PARAMETERS]
        jacobian = [[float(i+1) for i in range(8)], [float(-2*(i+1)) for i in range(8)]]
        (ad/"summary.json").write_text(json.dumps(dict(parameter_order=columns,
            residual_labels=["transport:"+label for label in labels],
            residuals=[2., 3.], jacobian=jacobian)))
        arrays = dict(initial_y=np.array([1., 2., 3.]), final_y=np.array([2., 3., 4.]),
            final_time=np.array(.1), **{"support['geometry'].r_grid": np.array([.1, .5, .9]),
            "support['database'].D11_log": np.array([-2., -1.])})
        np.savez(ad/"snapshots/ad_terminal_saved.npz", **arrays)
        for i, parameter in enumerate(runner.PARAMETERS):
            folder = root/f"fd_{parameter.replace(':', '_')}"
            (folder/"snapshots").mkdir(parents=True)
            (folder/"summary.json").write_text(json.dumps(dict(objective_labels=labels,
                objective_values=[2., 3.], gradient_fd=[jacobian[0][i], jacobian[1][i]])))
            np.savez(folder/"snapshots/fd_baseline_saved.npz", **arrays)
        return arrays

    def test_plan_prints_nine_commands_without_creating_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/"unused"
            result = subprocess.run([sys.executable, str(SCRIPT), "--out-dir", str(output), "--plan-only"],
                cwd=ROOT, capture_output=True, text=True, check=True)
            self.assertEqual(len(result.stdout.splitlines()), 9)
            self.assertIn("--accepted-step-limit 16", result.stdout)
            self.assertIn("--reverse-database-interpolation-transpose-mode established", result.stdout)
            self.assertFalse(output.exists())

    def test_matching_saved_arrays_produce_scientific_notation_comparisons(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.saved_case(root)
            with redirect_stdout(io.StringIO()):
                rows = runner.compare(root)
            self.assertEqual(len(rows), 16)
            self.assertTrue(all(row["comparison_baseline_matches"] for row in rows))
            self.assertTrue(all(row["relative_difference_to_fd"] == 0 for row in rows))
            with (root/"ad_vs_fd.csv").open() as stream:
                saved = list(csv.DictReader(stream))
            self.assertTrue(all("e" in row["relative_difference_to_fd"] for row in saved))

    def test_state_table_and_objective_baseline_mismatches_are_flagged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arrays = self.saved_case(root)
            folder = root/"fd_n0"
            changed = dict(arrays, initial_y=arrays["initial_y"]+.01)
            np.savez(folder/"snapshots/fd_baseline_saved.npz", **changed)
            folder = root/"fd_T0"
            changed = dict(arrays)
            changed["support['database'].D11_log"] = arrays["support['database'].D11_log"]+.001
            np.savez(folder/"snapshots/fd_baseline_saved.npz", **changed)
            folder = root/"fd_density_shape_power"
            report = json.loads((folder/"summary.json").read_text())
            report["objective_values"][0] += .1
            (folder/"summary.json").write_text(json.dumps(report))
            with redirect_stdout(io.StringIO()):
                rows = runner.compare(root)
            for row in rows:
                mismatch = row["parameter"] in ("n0", "T0") or (
                    row["parameter"] == "density_shape_power" and row["objective"] == "softmax_Er")
                self.assertEqual(row["comparison_baseline_matches"], not mismatch)


if __name__ == "__main__":
    unittest.main()
