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
    def test_saved_step_budget_is_preserved_unless_explicitly_overridden(self):
        saved = SimpleNamespace(
            summary={"config": {"transport_solver": {"max_steps": 2400}}},
            baseline_input=Path("/original/input.seed"),
        )
        for flags, expected in (([], 2400), (["--transport-max-steps", "3000"], 3000)):
            with self.subTest(flags=flags):
                args = example.parser().parse_args(["--geometry-result-dir", "/saved", *flags])
                example.configure_saved_settings(args, saved)
                self.assertEqual(args.transport_max_steps, expected)

    def test_modes_have_distinct_defaults_and_explicit_output_takes_precedence(self):
        arguments = ["--geometry-result-dir", "/saved"]
        combined_args = example.parser().parse_args(arguments)
        profile_args = example.parser().parse_args(arguments + ["--profile-only"])
        self.assertNotEqual(combined_args.out_dir, profile_args.out_dir)
        self.assertTrue(combined_args.plot_initial_profiles)
        self.assertTrue(profile_args.plot_initial_profiles)
        explicit = example.parser().parse_args(arguments + ["--profile-only", "--out-dir", "/explicit"])
        self.assertEqual(explicit.out_dir, Path("/explicit"))

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
                patch("sys.argv", [example.__file__, "--geometry-result-dir", str(saved), "--out-dir", str(output),
                                   "--transport-max-steps", "2000"]),
                patch.object(example.combined.geometry_example, "transport_config", return_value={"geometry": {"n_radial": 51}}),
                patch.object(example.combined, "_build_problem", side_effect=build),
                patch.object(example.combined, "report", return_value=evaluation),
                patch.object(example.warm, "MemoryDiagnosticProblem", side_effect=lambda problem, phase: problem),
                patch.object(example.warm, "_run_geometry_phase", side_effect=AssertionError("geometry phase must never run")) as geometry_phase,
                patch.object(example.opt, "least_squares", side_effect=solve),
                patch.object(example, "write_initial_profile_comparison", return_value={"png": "mock.png"}) as plot,
                patch.object(example.combined, "write_outputs", side_effect=AssertionError("postprocessing must never run")) as postprocess,
            ):
                self.assertEqual(example.main(), 0)
                geometry_phase.assert_not_called()
                postprocess.assert_not_called()
                plot.assert_called_once()

            self.assertEqual(captured["build"]["vmec_input"], baseline)
            self.assertEqual(captured["build"]["max_mode"], 2)
            self.assertEqual(captured["build"]["config"]["transport_solver"]["max_steps"], 2000)
            self.assertEqual(captured["switches"], (True, False))
            np.testing.assert_array_equal(captured["x0"], np.concatenate((np.zeros(6), accepted)))
            written = json.loads((output / "optimization_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(written["nfev"], 2)
            self.assertEqual(written["transport_max_steps"], 2000)
            self.assertEqual(written["geometry_result"], str(saved / "optimization_summary.json"))
            self.assertTrue((output / "input.QI_combined_optimized").is_file())
            self.assertTrue((output / "profile_parameters_optimized.json").is_file())
            self.assertTrue((output / "profile_parameters_initial.json").is_file())
            self.assertEqual(written["optimization_mode"], "geometry_and_profiles")


class ProfileOnlyTests(unittest.TestCase):
    def test_six_variable_scipy_solve_never_changes_saved_geometry(self):
        """Exercise the real adapter/SciPy with a cheap analytic residual."""
        profile_names = tuple(example.combined.PROFILE_PARAMETERS.split(","))
        nominal = np.asarray([4.21, 17.8, 10.0, 2.0, 1.0, 1.0])
        saved_geometry = np.asarray([.125, -.375])
        target = np.asarray([.01, -.02, .03, -.04, .05, -.06])
        calls = []

        class FullProblem:
            parameter_labels = profile_names + ("RBC:1:0", "ZBS:1:0")
            parameter_count = 8
            x0 = np.zeros(8)
            x_scale = np.concatenate((nominal, [.25, .5]))
            profile_coordinate_mode = "delta"

            def profile_values_from_scaled_parameters(self, x):
                return nominal * (1 + np.asarray(x)[:6])

            def input_from_scaled_parameters(self, x):
                return np.asarray(x)[6:].copy()

            def evaluate(self, x):
                x = np.asarray(x)
                calls.append(x.copy())
                np.testing.assert_array_equal(x[6:], saved_geometry)
                residual = np.concatenate((x[:6] - target, [.4]))
                jacobian = np.concatenate((np.eye(6), np.zeros((1, 6))))
                # Nonzero geometry sensitivities must never reach SciPy.
                jacobian = np.column_stack((jacobian, np.ones((7, 2))))
                result = example.opt.LeastSquaresResult(
                    residuals=residual, jacobian=jacobian,
                    residual_labels=tuple(f"row_{i}" for i in range(7)),
                    parameter_labels=self.parameter_labels, objective_values={},
                )
                return example.opt.LeastSquaresEvaluation(
                    result=result, residuals=residual, jacobian=jacobian, elapsed_s=0,
                )

        restricted = example.ProfileOnlyProblem(FullProblem(), np.concatenate((np.zeros(6), saved_geometry)))
        initial = restricted.evaluate(restricted.x0)
        self.assertEqual(initial.jacobian.shape, (7, 6))
        self.assertEqual(initial.result.parameter_labels, profile_names)
        self.assertEqual(initial.result.jacobian.shape, (7, 6))
        with tempfile.TemporaryDirectory() as directory:
            solved = example.opt.least_squares(
                restricted, x0=restricted.x0, initial_evaluation=initial,
                bounds=example.combined.scaled_bounds(restricted), max_nfev=12,
                diagnostics_dir=directory,
            )
            events_path = next(Path(directory).glob("run_*/events.jsonl"))
            setup = json.loads(events_path.read_text().splitlines()[0])
            self.assertEqual(setup["parameter_labels"], list(profile_names))
        np.testing.assert_allclose(solved.x, target, atol=1e-7)
        np.testing.assert_array_equal(restricted.input_from_scaled_parameters(solved.x), saved_geometry)
        np.testing.assert_array_equal(restricted.full_parameters(solved.x)[6:], saved_geometry)
        self.assertGreater(len(calls), 1)
        with self.assertRaisesRegex(ValueError, "profile coordinates"):
            restricted.full_parameters(np.zeros(8))

    def test_profile_only_main_exposes_six_variables_and_saves_full_geometry(self):
        captured = {}
        profile_names = tuple(example.combined.PROFILE_PARAMETERS.split(","))
        nominal = np.asarray([4.21, 17.8, 10., 2., 1., 1.])
        accepted = np.asarray([.125, -.375])

        class Input:
            def __init__(self, values):
                self.values = np.asarray(values).copy()

            def to_indata(self, path):
                np.testing.assert_array_equal(self.values[6:], accepted)
                Path(path).write_text("fixed geometry")

        class Problem:
            parameter_labels = profile_names + ("RBC:1:0", "ZBS:1:0")
            parameter_count = 8
            x0 = np.zeros(8)
            x_scale = np.concatenate((nominal, [.25, .5]))
            profile_coordinate_mode = "delta"

            def input_from_scaled_parameters(self, x):
                return Input(x)

            def profile_values_from_scaled_parameters(self, x):
                return nominal * (1 + np.asarray(x)[:6])

        initial = SimpleNamespace(residuals=np.ones(1))

        def solve(problem, *, x0, **kwargs):
            captured["labels"] = problem.parameter_labels
            np.testing.assert_array_equal(x0, np.zeros(6))
            self.assertEqual(kwargs["bounds"][0].shape, (6,))
            self.assertEqual(kwargs["x_scale"].shape, (6,))
            return SimpleNamespace(x=np.full(6, .001), cost=.25, nfev=2, status=1, message="mock")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            saved = root / "saved"
            saved.mkdir()
            baseline = root / "input.seed"
            baseline.write_text("seed")
            (saved / "optimization_summary.json").write_text(json.dumps({
                "seed_input": str(baseline), "max_mode_schedule": [2],
                "parameter_labels": ["RBC:1:0", "ZBS:1:0"], "x": accepted.tolist(),
            }))
            output = root / "profiles"
            with (
                patch("sys.argv", [example.__file__, "--geometry-result-dir", str(saved),
                                   "--profile-only", "--no-plot-initial-profiles", "--out-dir", str(output)]),
                patch.object(example.combined.geometry_example, "transport_config", return_value={"geometry": {"n_radial": 51}}),
                patch.object(example.combined, "_build_problem", return_value=Problem()),
                patch.object(example.combined, "report", return_value=initial),
                patch.object(example.warm, "MemoryDiagnosticProblem", side_effect=lambda problem, phase: problem),
                patch.object(example.opt, "least_squares", side_effect=solve),
                patch.object(example, "write_initial_profile_comparison") as plot,
            ):
                self.assertEqual(example.main(), 0)
                plot.assert_not_called()
            self.assertEqual(captured["labels"], profile_names)
            summary = json.loads((output / "optimization_summary.json").read_text())
            self.assertEqual(summary["optimization_mode"], "profiles_only")
            self.assertEqual(len(summary["x"]), 6)
            np.testing.assert_array_equal(summary["full_x"][6:], accepted)
            np.testing.assert_array_equal(summary["full_x0"][6:], accepted)


class TrialReplayTests(unittest.TestCase):
    def test_failed_trial_coordinates_are_evaluated_once_without_optimizer_or_nominal_report(self):
        profiles = np.asarray([4.21, 17.8, 10., 2., 1., 1.])
        labels = tuple(example.combined.PROFILE_PARAMETERS.split(","))
        geometry = np.asarray([.125, -.375])
        trial_x = np.asarray([.001, -.002, .003, -.004, .005, -.006])
        calls = []

        class Input:
            def to_indata(self, path):
                Path(path).write_text("fixed geometry")

        class Problem:
            parameter_labels = labels + ("RBC:1:0", "ZBS:1:0")
            parameter_count = 8
            x0 = np.zeros(8)
            x_scale = np.concatenate((profiles, [.25, .5]))
            baseline_profile_values = profiles
            profile_coordinate_mode = "delta"

            def input_from_scaled_parameters(self, x):
                np.testing.assert_array_equal(np.asarray(x)[6:], geometry)
                return Input()

            def profile_values_from_scaled_parameters(self, x):
                return profiles * (1 + np.asarray(x)[:6])

            def evaluate(self, x):
                calls.append(np.asarray(x).copy())
                residual, jacobian = np.ones(1), np.ones((1, 8))
                result = example.opt.LeastSquaresResult(
                    residuals=residual, jacobian=jacobian, residual_labels=("row",),
                    parameter_labels=self.parameter_labels, objective_values={},
                )
                return example.opt.LeastSquaresEvaluation(result=result, residuals=residual,
                                                           jacobian=jacobian, elapsed_s=0.)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            saved, source = root / "geometry", root / "old_diagnostics"
            output = root / "replay"
            saved.mkdir()
            source.mkdir()
            (saved / "optimization_summary.json").write_text(json.dumps({
                "seed_input": str(root / "seed"), "max_mode_schedule": [2],
                "parameter_labels": ["RBC:1:0", "ZBS:1:0"], "x": geometry.tolist(),
            }))
            events = [
                {"event": "setup", "parameter_labels": labels, "physical_scales": profiles.tolist(),
                 "baseline_profile_values": profiles.tolist(), "profile_coordinate_mode": "delta", "trust_scales": [1.] * 6},
                {"event": "failed_trial", "trial": 2, "x": trial_x.tolist(), "reason": "old nonfinite scan bars"},
            ]
            original = "".join(json.dumps(event) + "\n" for event in events)
            (source / "events.jsonl").write_text(original)
            with (
                patch("sys.argv", [example.__file__, "--geometry-result-dir", str(saved), "--profile-only",
                                   "--replay-diagnostics", str(source), "--replay-trial", "2", "--out-dir", str(output)]),
                patch.object(example.combined.geometry_example, "transport_config", return_value={"geometry": {"n_radial": 51}}),
                patch.object(example.combined.geometry_example, "iteration_diagnostics", return_value="mock"),
                patch.object(example.combined, "_build_problem", return_value=Problem()),
                patch.object(example.combined, "report", side_effect=AssertionError("no nominal report")) as nominal,
                patch.object(example.warm, "MemoryDiagnosticProblem", side_effect=lambda problem, phase: problem),
                patch.object(example.opt, "least_squares", side_effect=AssertionError("no optimizer")) as solve,
                patch.object(example, "write_initial_profile_comparison", side_effect=AssertionError("no plots")) as plot,
            ):
                self.assertEqual(example.main(), 0)
                solve.assert_not_called()
                nominal.assert_not_called()
                plot.assert_not_called()
            self.assertEqual(len(calls), 1)
            np.testing.assert_array_equal(calls[0], np.concatenate((trial_x, geometry)))
            report = json.loads((output / "replay_summary.json").read_text())
            self.assertTrue(report["success"])
            self.assertEqual(report["physics_evaluations"], 1)
            self.assertEqual(report["source_trial"], 2)
            self.assertEqual(report["jacobian_shape"], [1, 6])
            self.assertEqual((source / "events.jsonl").read_text(), original)
            self.assertFalse((output / "optimization_summary.json").exists())

    def test_replay_error_is_preserved_instead_of_becoming_a_penalty(self):
        labels = tuple(example.combined.PROFILE_PARAMETERS.split(","))

        class Problem:
            parameter_count = 6
            parameter_labels = labels
            profile_coordinate_mode = "delta"
            x_scale = np.ones(6)
            x0 = np.zeros(6)

            def profile_values_from_scaled_parameters(self, x):
                return np.asarray([4.21, 17.8, 10., 2., 1., 1.])

            def evaluate(self, x):
                raise ValueError("test bad stage transpose")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            source.write_text("".join(json.dumps(event) + "\n" for event in [
                {"event": "setup", "parameter_labels": labels, "physical_scales": [1.] * 6,
                 "profile_coordinate_mode": "delta"},
                {"event": "failed_trial", "trial": 2, "x": [0.] * 6},
            ]))
            with self.assertRaisesRegex(ValueError, "bad stage transpose"):
                example.replay_saved_trial(Problem(), source=source, trial=2, out_dir=root / "output")
            report = json.loads((root / "output" / "replay_summary.json").read_text())
            self.assertFalse(report["success"])
            self.assertEqual(report["reason"], "test bad stage transpose")


class InitialProfilePlotTests(unittest.TestCase):
    def test_legacy_completed_run_can_plot_without_building_or_optimizing(self):
        """A real plot/CSV from saved scalar settings; no physics builder runs."""
        config = {
            "geometry": {"n_radial": 51}, "species": {"n_species": 4, "names": ["e", "D", "T", "He"]},
            "profiles": {"n0": 4.21, "T0": 17.8, "n_edge": .6, "T_edge": .7,
                         "density_shape_power": 10., "temperature_shape_power": 2.,
                         "c_density": [1., .5, .5, 0.], "n_scale": [.65, .65, .65, 0.], "T_scale": .7},
        }
        optimized = dict(zip(example.combined.PROFILE_PARAMETERS.split(","), [4.25, 17.7, 10., 2., 1., 1.]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            saved, output = root / "geometry", root / "completed"
            saved.mkdir()
            output.mkdir()
            (saved / "optimization_summary.json").write_text(json.dumps({
                "seed_input": str(root / "unavailable_seed"), "max_mode_schedule": [2],
                "parameter_labels": ["RBC:1:0"], "x": [.25], "config": config,
            }))
            result_path = output / "optimization_summary.json"
            old_text = json.dumps({"optimized_physical_profiles": optimized, "x": [0.] * 30})
            result_path.write_text(old_text)
            with (
                patch("sys.argv", [example.__file__, "--geometry-result-dir", str(saved),
                                   "--out-dir", str(output), "--plot-profiles-only"]),
                patch.object(example.combined.geometry_example, "transport_config", return_value=config),
                patch.object(example.combined, "_build_problem", side_effect=AssertionError("no physics build")) as build,
                patch.object(example.opt, "least_squares", side_effect=AssertionError("no optimization")) as solve,
            ):
                self.assertEqual(example.main(), 0)
                build.assert_not_called()
                solve.assert_not_called()
            self.assertTrue((output / "initial_profiles_initial_vs_optimized.png").is_file())
            data = np.genfromtxt(output / "initial_optimized_initial_profiles.csv", delimiter=",", names=True)
            self.assertEqual(data.shape, (51,))
            self.assertTrue(np.all(data["optimized_e_density_1e20_m3"] > data["initial_e_density_1e20_m3"]))
            self.assertEqual(result_path.read_text(), old_text)

    def test_plot_and_csv_use_initial_map_with_finite_zero_density_helium_temperature(self):
        """Actual small analytical-profile map; no root, VMEC, or transport call."""
        import jax.numpy as jnp
        from NEOPAX._state import TransportState
        from NEOPAX import _reverse_ad_transport as reverse

        rho = jnp.asarray([.1, .3, .5, .7, .9])
        geometry = SimpleNamespace(r_grid=rho, r_grid_half=jnp.asarray([0., 1.]), rho_grid=rho)
        nominal = np.asarray([4.21, 17.8, 10., 2., 1., 1.])
        profile_cfg = {
            "n0": 4.21, "T0": 17.8, "n_edge": .6, "T_edge": .7,
            "density_shape_power": 10., "temperature_shape_power": 2.,
            "c_density": [1., .5, .5, 0.], "n_scale": [.65, .65, .65, 0.],
            "T_scale": .7, "c_temperature": [1., 1., 1., 1.],
        }

        class Problem:
            config = {"profiles": profile_cfg, "species": {"names": ["e", "D", "T", "He"]},
                      "transport_solver": {"density_floor": 1e-6, "temperature_floor": 1e-6}}
            runtime = SimpleNamespace(geometry=geometry, species=SimpleNamespace(number_species=4))
            baseline_state = TransportState(density=jnp.ones((4, 5)), pressure=jnp.ones((4, 5)), Er=jnp.zeros(5))

            def profile_values_from_scaled_parameters(self, x):
                return nominal * (1 + np.asarray(x))

        problem = Problem()
        initial_x, optimized_x = np.zeros(6), np.asarray([.01, -.02, 0, 0, 0, 0])
        with (
            patch.object(reverse, "state_with_initial_er_root_ad", side_effect=AssertionError("must not solve initial Er")),
            tempfile.TemporaryDirectory() as directory,
        ):
            output = example.write_initial_profile_comparison(problem, initial_x, optimized_x, Path(directory))
            self.assertTrue(Path(output["png"]).stat().st_size > 1000)
            import csv
            with Path(output["csv"]).open() as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 5)
            for i, row in enumerate(rows):
                expected_t = .7 * ((17.8 - .7) * (1 - float(rho[i]) ** 2) + .7)
                self.assertAlmostEqual(float(row["initial_He_temperature_keV"]), expected_t, places=10)
                self.assertEqual(float(row["initial_He_density_1e20_m-3"]), 0)
                self.assertGreater(float(row["optimized_e_density_1e20_m-3"]), float(row["initial_e_density_1e20_m-3"]))
                self.assertLess(float(row["optimized_e_temperature_keV"]), float(row["initial_e_temperature_keV"]))


if __name__ == "__main__":
    unittest.main()
