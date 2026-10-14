"""Lightweight solver regressions; no VMEC, NTX, or transport solve is run."""

from types import SimpleNamespace
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from NEOPAX import optimization


class ScalarProblem:
    parameter_count = 1
    x0 = np.asarray([0.0])

    def __init__(self):
        self.calls = []

    def evaluate(self, values):
        values = np.asarray(values, dtype=float)
        self.calls.append(values.copy())
        return SimpleNamespace(
            residuals=values - 3.0,
            jacobian=np.ones((1, 1)),
        )


class SolverLifecycleTests(unittest.TestCase):
    def test_default_start_is_unchanged(self):
        problem = ScalarProblem()

        def scipy_check(fun, x0, *, jac, **kwargs):
            np.testing.assert_array_equal(x0, problem.x0)
            np.testing.assert_array_equal(kwargs["x_scale"], np.ones(1))
            return SimpleNamespace(x=x0)

        with patch("scipy.optimize.least_squares", scipy_check):
            optimization.least_squares(problem)

    def test_invalid_start_is_rejected_before_scipy(self):
        for start in ([1.0, 2.0], [np.nan], [np.inf]):
            with self.subTest(start=start):
                with patch("scipy.optimize.least_squares") as scipy:
                    with self.assertRaisesRegex(ValueError, "finite scaled coordinates"):
                        optimization.least_squares(ScalarProblem(), x0=start)
                    scipy.assert_not_called()

    def test_explicit_start_and_cached_evaluation_use_the_same_coordinates(self):
        problem = ScalarProblem()
        start = np.asarray([2.5])
        initial = problem.evaluate(start)

        def scipy_check(fun, x0, *, jac, **kwargs):
            np.testing.assert_array_equal(x0, start)
            np.testing.assert_array_equal(fun(x0), [-0.5])
            np.testing.assert_array_equal(jac(x0), [[1.0]])
            self.assertEqual(len(problem.calls), 1)
            np.testing.assert_array_equal(fun(np.asarray([2.75])), [-0.25])
            np.testing.assert_array_equal(problem.calls[-1], [2.75])
            return SimpleNamespace(x=x0)

        with patch("scipy.optimize.least_squares", scipy_check):
            optimization.least_squares(
                problem, x0=start, initial_evaluation=initial
            )

    def test_small_warm_start_solve_reaches_the_actual_minimum(self):
        problem = ScalarProblem()
        start = np.asarray([2.5])
        result = optimization.least_squares(
            problem, x0=start, initial_evaluation=problem.evaluate(start)
        )
        np.testing.assert_allclose(result.x, [3.0], atol=1e-12)
        self.assertLess(result.cost, 1e-20)

    def test_resource_exhaustion_is_fatal_in_residual_and_jacobian_callbacks(self):
        for callback in ("fun", "jac"):
            for error in (
                RuntimeError("RESOURCE_EXHAUSTED: CUDA graph instantiation"),
                ValueError("CUDA_ERROR_OUT_OF_MEMORY: out of memory"),
                MemoryError("host allocation failed"),
            ):
                with self.subTest(callback=callback, error=error):
                    problem = ScalarProblem()
                    initial = problem.evaluate(problem.x0)

                    def fail(_values):
                        raise error

                    problem.evaluate = fail

                    def scipy_check(fun, x0, *, jac, **kwargs):
                        np.testing.assert_array_equal(fun(x0), [-3.0])
                        (fun if callback == "fun" else jac)(np.asarray([1.0]))
                        self.fail("Resource exhaustion was hidden as a trial penalty")

                    with patch("scipy.optimize.least_squares", scipy_check):
                        with self.assertRaises(type(error)):
                            optimization.least_squares(
                                problem, initial_evaluation=initial
                            )

    def test_nonconverged_trial_still_uses_the_existing_penalty(self):
        problem = ScalarProblem()
        initial = problem.evaluate(problem.x0)

        def fail(_values):
            raise RuntimeError("VMEC did not converge")

        problem.evaluate = fail

        def scipy_check(fun, x0, *, jac, **kwargs):
            np.testing.assert_array_equal(fun(x0), [-3.0])
            np.testing.assert_array_equal(fun(np.asarray([1.0])), [1e6])
            np.testing.assert_array_equal(jac(np.asarray([1.0])), [[0.0]])
            return SimpleNamespace(x=x0)

        with patch("scipy.optimize.least_squares", scipy_check):
            optimization.least_squares(problem, initial_evaluation=initial)


class GeometryPhaseTests(unittest.TestCase):
    def test_geometry_phase_matches_standalone_values_and_solver_options(self):
        from examples.optimization import (
            optimize_geometry_warm_start_profiles_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport
            as warm,
        )

        geometry = warm.geometry_example
        evaluation = SimpleNamespace(residuals=np.asarray([1.0]))
        config = {"geometry": {"n_radial": 51}}
        builds, solves = [], []

        class Input:
            def to_indata(self, path):
                Path(path).write_text("mock VMEC input", encoding="utf-8")

        class Problem:
            parameter_count = 2
            parameter_labels = ("RBC:1:0", "ZBS:1:0")
            x0 = np.zeros(2)
            x_scale = np.asarray([0.25, 0.5])

            def input_from_scaled_parameters(self, values):
                return Input()

        def build(config, terms, **kwargs):
            normalized = optimization._normalize_initial_er_root_least_squares_terms(terms)
            term_values = tuple(
                (term.objective.family, term.objective.name, term.target, term.weight, term.label)
                for term in normalized
            )
            builds.append((config, term_values, kwargs))
            return Problem()

        def solve(problem, **kwargs):
            solves.append((np.asarray(problem.x0), kwargs))
            return SimpleNamespace(
                x=np.asarray([0.125, -0.375]), cost=0.5, optimality=0.0,
                nfev=2, status=1, message="mock convergence",
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = warm.parser().parse_args([
                "--seed-input", str(root / "input.seed"),
                "--out-dir", str(root / "standalone"),
                "--max-er", "--root-objectives", "--no-transition-location-objective",
            ])
            with (
                patch.object(geometry, "parser", return_value=SimpleNamespace(parse_args=lambda: args)),
                patch.object(geometry, "transport_config", return_value=config),
                patch.object(geometry, "max_mode_schedule_values", return_value=(2,)),
                patch.object(geometry, "report", return_value=evaluation),
                patch.object(geometry, "write_outputs"),
                patch.object(optimization, "geometry_full_transport_least_squares_problem", side_effect=build),
                patch.object(optimization, "least_squares", side_effect=solve),
            ):
                geometry.main()
                phase = warm._run_geometry_phase(
                    args, config=config, seed_input=args.seed_input.resolve(),
                    out_dir=root, max_modes=(2,), max_nfev=args.max_nfev,
                    physical_pitches=geometry.PHYSICAL_J_PITCHES,
                )

            self.assertEqual(builds[0], builds[1])
            np.testing.assert_array_equal(solves[0][0], solves[1][0])
            self.assertEqual(solves[0][1], solves[1][1])
            np.testing.assert_array_equal(phase["result"].x, [0.125, -0.375])
            self.assertFalse(hasattr(phase["coordinates"], "evaluate"))
            np.testing.assert_array_equal(phase["coordinates"].x_scale, Problem.x_scale)

    def test_memory_diagnostics_observe_one_delegated_evaluation(self):
        from examples.optimization import (
            optimize_geometry_warm_start_profiles_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport
            as warm,
        )

        problem = ScalarProblem()
        problem.table_result_builder = SimpleNamespace(
            optimization_runtime_scan_cache_size=lambda: 1,
        )
        wrapped = warm.MemoryDiagnosticProblem(problem, "combined_m2")
        output = io.StringIO()
        device = SimpleNamespace(memory_stats=lambda: {"bytes_in_use": 1024})
        with (
            patch.object(optimization, "_process_resident_memory_bytes", return_value=2**20),
            patch.object(warm.jax, "live_arrays", return_value=[None, None]),
            patch.object(warm.jax, "devices", return_value=[device]),
            redirect_stdout(output),
        ):
            evaluation = wrapped.evaluate(np.asarray([2.5]))

        self.assertEqual(len(problem.calls), 1)
        np.testing.assert_array_equal(evaluation.residuals, [-0.5])
        self.assertIn("phase=combined_m2 evaluation=1 rss_MiB=1.0", output.getvalue())
        self.assertIn("live_jax_arrays=2", output.getvalue())
        self.assertIn("device_bytes_in_use=1024", output.getvalue())
        self.assertIn("'runtime_scan': 1", output.getvalue())

    def test_combined_main_passes_accepted_geometry_to_the_actual_solver(self):
        from examples.optimization import (
            optimize_geometry_warm_start_profiles_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport
            as warm,
            optimize_geometry_profiles_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_simple_least_squares
            as combined,
        )

        geometry_coordinates = SimpleNamespace(
            parameter_labels=("RBC:1:0", "ZBS:1:0"),
            x_scale=np.asarray([0.25, 0.5]),
        )
        accepted = np.asarray([0.125, -0.375])
        profile_values = np.asarray([4.21, 17.8, 10.0, 2.0, 1.0, 1.0])

        class Input:
            def __init__(self, geometry):
                self.geometry = np.asarray(geometry)

            def to_indata(self, path):
                Path(path).write_text(repr(self.geometry.tolist()), encoding="utf-8")

        class Problem:
            parameter_labels = tuple(combined.PROFILE_PARAMETERS.split(",")) + geometry_coordinates.parameter_labels
            parameter_count = len(parameter_labels)
            x_scale = np.concatenate((profile_values, geometry_coordinates.x_scale))
            x0 = np.zeros(parameter_count)
            profile_coordinate_mode = "delta"

            def input_from_scaled_parameters(self, x):
                return Input(np.asarray(x)[-2:])

            def profile_values_from_scaled_parameters(self, x):
                return profile_values * (1.0 + np.asarray(x)[:6])

        class StopBeforeSolver(Exception):
            pass

        evaluation = SimpleNamespace(residuals=np.asarray([1.0]))

        def check_start(problem, **kwargs):
            np.testing.assert_array_equal(kwargs["x0"], np.concatenate((np.zeros(6), accepted)))
            self.assertIs(kwargs["initial_evaluation"], evaluation)
            raise StopBeforeSolver

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            warm_path = root / "input.warm"
            Input(accepted).to_indata(warm_path)
            phase = {
                "baseline_input": root / "input.seed", "physical_pitches": None,
                "coordinates": geometry_coordinates, "initial_cost": 1.0,
                "result": SimpleNamespace(x=accepted, cost=0.5, nfev=2,
                                          status=1, message="mock convergence"),
                "warm_path": warm_path,
            }
            with (
                patch("sys.argv", [warm.__file__, "--out-dir", directory]),
                patch.object(warm, "_run_geometry_phase", return_value=phase),
                patch.object(warm.geometry_example, "transport_config", return_value={"geometry": {"n_radial": 51}}),
                patch.object(combined, "_build_problem", return_value=Problem()),
                patch.object(combined, "report", return_value=evaluation),
                patch.object(optimization, "least_squares", side_effect=check_start),
            ):
                with self.assertRaises(StopBeforeSolver):
                    warm.main()


if __name__ == "__main__":
    unittest.main()
