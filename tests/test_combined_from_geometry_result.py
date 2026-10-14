"""Mocked checks for saved-result continuation; no plasma physics is run."""

import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from examples.optimization import optimize_geometry_profiles_from_geometry_result as example


class SavedGeometryTests(unittest.TestCase):
    def test_cuda_graph_flag_is_configured_before_standalone_imports(self):
        with patch.dict(os.environ, {"XLA_FLAGS": "--existing_flag=value"}):
            example._configure_command_buffers([])
            self.assertEqual(os.environ["XLA_FLAGS"], "--existing_flag=value --xla_gpu_enable_command_buffer=")
        with patch.dict(os.environ, {"XLA_FLAGS": "--existing_flag=value"}):
            example._configure_command_buffers(["--enable-cuda-command-buffers"])
            self.assertEqual(os.environ["XLA_FLAGS"], "--existing_flag=value")

    def test_saved_continuation_uses_last_stage_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            summary = {
                "seed_input": "/original/input.seed", "max_mode_schedule": [1, 2],
                "parameter_labels": ["RBC:1:0", "ZBS:1:0"], "x": [0.125, -0.375],
            }
            (path / "optimization_summary.json").write_text(json.dumps(summary), encoding="utf-8")
            result = example.load_geometry_result(path)
            self.assertEqual(result.max_mode, 2)
            self.assertEqual(result.baseline_input, path / "input.QI_neopax_database_full_transport_net_power_stage_m1")

    def test_existing_geometry_result_runs_only_combined_solver(self):
        profiles = np.asarray([4.21, 17.8, 10.0, 2.0, 1.0, 1.0])
        geometry_labels = ("RBC:1:0", "ZBS:1:0")
        accepted = np.asarray([0.125, -0.375])
        captured = {}

        class Input:
            def __init__(self, x):
                self.x = np.asarray(x)

            def to_indata(self, path):
                Path(path).write_text(repr(self.x[-2:].tolist()), encoding="utf-8")

        class Problem:
            parameter_labels = tuple(example.combined.PROFILE_PARAMETERS.split(",")) + geometry_labels
            parameter_count = len(parameter_labels)
            x0 = np.zeros(parameter_count)
            x_scale = np.concatenate((profiles, [0.25, 0.5]))
            profile_coordinate_mode = "delta"

            def input_from_scaled_parameters(self, x):
                return Input(x)

            def profile_values_from_scaled_parameters(self, x):
                return profiles * (1.0 + np.asarray(x)[:6])

        evaluation = SimpleNamespace(residuals=np.asarray([1.0]))

        def build(args, **kwargs):
            captured["build"] = kwargs
            captured["switches"] = (args.root_objectives, args.transition_location_objective)
            return Problem()

        def solve(problem, *, x0, **kwargs):
            captured["x0"] = x0.copy()
            self.assertIs(kwargs["initial_evaluation"], evaluation)
            return SimpleNamespace(x=x0.copy(), cost=0.5, nfev=2, status=1, message="mock convergence")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            saved = root / "saved"
            saved.mkdir()
            baseline = root / "input.seed"
            baseline.write_text("mock baseline", encoding="utf-8")
            summary = {
                "seed_input": str(baseline), "max_mode_schedule": [2],
                "parameter_labels": list(geometry_labels), "x": accepted.tolist(),
                "database_grid": [5, 25, 31],
                "objectives": {"root_objectives": True, "transition_location": False,
                               "bootstrap_penalty": True, "net_power": True},
            }
            (saved / "optimization_summary.json").write_text(json.dumps(summary), encoding="utf-8")
            output = root / "combined"
            with (
                patch("sys.argv", [example.__file__, "--geometry-result-dir", str(saved), "--out-dir", str(output)]),
                patch.object(example.combined.geometry_example, "transport_config", return_value={"geometry": {"n_radial": 51}}),
                patch.object(example.combined, "_build_problem", side_effect=build),
                patch.object(example.combined, "report", return_value=evaluation),
                patch.object(example.warm, "MemoryDiagnosticProblem", side_effect=lambda problem, phase: problem),
                patch.object(example.warm, "_run_geometry_phase", side_effect=AssertionError("geometry phase must never run")) as geometry_phase,
                patch.object(example.opt, "least_squares", side_effect=solve),
                patch.object(example.combined, "write_outputs", side_effect=AssertionError("postprocessing must never run")) as postprocess,
            ):
                self.assertEqual(example.main(), 0)
                geometry_phase.assert_not_called()
                postprocess.assert_not_called()

            self.assertEqual(captured["build"]["vmec_input"], baseline)
            self.assertEqual(captured["build"]["max_mode"], 2)
            self.assertEqual(captured["switches"], (True, False))
            np.testing.assert_array_equal(captured["x0"], np.concatenate((np.zeros(6), accepted)))
            written = json.loads((output / "optimization_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(written["nfev"], 2)
            self.assertEqual(written["geometry_result"], str(saved / "optimization_summary.json"))
            self.assertTrue((output / "input.QI_combined_optimized").is_file())
            self.assertTrue((output / "profile_parameters_optimized.json").is_file())


if __name__ == "__main__":
    unittest.main()
