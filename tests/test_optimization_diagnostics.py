"""Small analytic/mocked checks; never run VMEC, NTX, or transport physics."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from NEOPAX import optimization
from NEOPAX._optimization_diagnostics import (
    LeastSquaresDiagnostics, reverse_checkpoint, reverse_diagnostics_active,
    reverse_objective_values,
)


class MixedProblem:
    parameter_count = 2
    parameter_labels = ("n0", "RBC:1:0")
    x0 = np.zeros(2)
    x_scale = np.asarray([4.0, 0.5])

    def __init__(self):
        self.calls = []

    def evaluate(self, x):
        import jax.numpy as jnp
        self.calls.append(np.asarray(x).copy())
        if x[0] == -99:
            reverse_objective_values(np.asarray([.52, 300.]), ("root", "power"))
            reverse_checkpoint("segment_1_support_increment", {
                "D11_log": jnp.asarray([[1., 2.], [np.nan, 3.]])
            }, objective_labels=("root", "power"))
            raise FloatingPointError("nonfinite table cotangent")
        jac = np.asarray([[2., 3.], [-4., 5.]])
        residual = np.asarray([1., 2.]) + jac @ np.asarray(x)
        return SimpleNamespace(residuals=residual, jacobian=jac, result=SimpleNamespace(
            residual_labels=("root", "power"), objective_values={"root": jnp.asarray(residual[0])},
        ))


def records(root):
    runs = list(Path(root).glob("run_*"))
    if len(runs) != 1:
        raise AssertionError(f"Expected one diagnostics directory, got {runs}")
    return runs[0], [json.loads(line) for line in (runs[0] / "events.jsonl").read_text().splitlines()]


class OptimizationDiagnosticsTests(unittest.TestCase):
    def test_rejected_failed_and_accepted_trials_and_exact_model_are_distinguished(self):
        problem = MixedProblem()
        initial = problem.evaluate(problem.x0)

        def solve(fun, x0, *, jac, **kwargs):
            self.assertNotIn("diagnostics_dir", kwargs)
            np.testing.assert_array_equal(fun(x0), [1., 2.])
            jac(x0)
            np.testing.assert_array_equal(fun([-99., .5]), [1e6, 1e6])
            np.testing.assert_array_equal(jac([-99., .5]), np.zeros((2, 2)))
            fun([1., 1.])  # Rejected: no Jacobian request at this point.
            fun([.1, -.1])
            jac([.1, -.1])  # Accepted by SciPy.
            return SimpleNamespace(x=np.asarray([.1, -.1]), cost=1.01, status=0,
                                   nfev=4, njev=2, optimality=1.0, message="budget exhausted")

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            with patch("scipy.optimize.least_squares", solve):
                optimization.least_squares(problem, initial_evaluation=initial,
                                           diagnostics_dir=directory, x_scale=[2., 1.])
            run, events = records(directory)
            self.assertEqual(len(problem.calls), 4)  # Initial + three trials, no extra calls.
            self.assertEqual([e["trial"] for e in events if e["event"] == "initial"], [1])
            self.assertEqual([e["trial"] for e in events if e["event"] == "accepted"], [4])
            failed = next(e for e in events if e["event"] == "failed_trial")
            self.assertEqual(failed["trial"], 2)
            self.assertEqual(failed["first_nonfinite_boundary"], "segment_1_support_increment")
            self.assertEqual(failed["transport_objective_values"], {"root": .52, "power": 300.})
            self.assertIn("FloatingPointError", (run / failed["traceback"]).read_text())
            bad = next(e for e in events if e["event"] == "reverse_checkpoint")
            self.assertEqual(bad["nonfinite"][0]["objectives"], ["power"])
            self.assertEqual(bad["nonfinite"][0]["first_nonfinite_index"], [1, 0])
            trial = next(e for e in events if e["event"] == "trial" and e["trial"] == 4)
            np.testing.assert_allclose(trial["model"]["predicted_cost"], [.405, .605])
            self.assertAlmostEqual(trial["model"]["actual_over_predicted"], 1.)
            self.assertAlmostEqual(trial["model"]["rows"][0]["geometry_linear_change"], -.3)
            self.assertAlmostEqual(trial["model"]["rows"][0]["profile_linear_change"], .2)
            np.testing.assert_allclose(trial["model"]["physical_step"], [.4, -.05])
            snapshot = np.load(run / trial["snapshot"])
            np.testing.assert_array_equal(snapshot["jacobian"], initial.jacobian)
            np.testing.assert_allclose(snapshot["residuals"], [.9, 1.1])
            np.testing.assert_allclose(trial["rows"][0]["gradient_profile"], [1.8])
            np.testing.assert_allclose(trial["rows"][0]["gradient_geometry"], [2.7])
            self.assertEqual(events[-1]["accepted_trials"], [1, 4])
            self.assertFalse(reverse_diagnostics_active())

    def test_real_bounded_scipy_solve_has_identical_trajectory_and_evaluation_count(self):
        class ScalarProblem:
            parameter_count = 1
            parameter_labels = ("RBC:1:0",)
            x_scale = np.ones(1)
            x0 = np.zeros(1)

            def __init__(self):
                self.calls = []

            def evaluate(self, x):
                self.calls.append(np.asarray(x).copy())
                return SimpleNamespace(residuals=np.asarray(x) - 3., jacobian=np.ones((1, 1)))

        ordinary, diagnosed = ScalarProblem(), ScalarProblem()
        expected = optimization.least_squares(ordinary, bounds=([-1.], [1.]))
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            actual = optimization.least_squares(diagnosed, bounds=([-1.], [1.]), diagnostics_dir=directory)
            _, events = records(directory)
            np.testing.assert_array_equal(actual.x, expected.x)
            np.testing.assert_array_equal(ordinary.calls, diagnosed.calls)
            self.assertEqual(actual.nfev, expected.nfev)
            self.assertEqual(len([e for e in events if e["event"] in ("initial", "accepted")]), actual.njev)
            self.assertEqual(events[-1]["active_mask"], [1])

    def test_state_floor_summaries_and_scope_reset_without_retaining_device_arrays(self):
        import jax.numpy as jnp
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            sink = LeastSquaresDiagnostics(directory, MixedProblem())
            with sink.evaluation_scope():
                self.assertTrue(reverse_diagnostics_active())
                density = jnp.asarray([[1., 2.], [0., 0.]])
                reverse_checkpoint("initial_state", {"density": density}, summarize=True)
                reverse_checkpoint("selected_root", {"mask": jnp.asarray([True, False])}, summarize=True)
                reverse_checkpoint("root", {"bars": jnp.asarray([1., np.inf])})
                reverse_checkpoint("later", {"bars": jnp.asarray([np.nan])})
            self.assertFalse(reverse_diagnostics_active())
            reverse_checkpoint("disabled", object())
            _, events = records(directory)
            checkpoints = [e for e in events if e["event"] == "reverse_checkpoint"]
            self.assertEqual(len(checkpoints), 4)
            self.assertEqual(checkpoints[0]["summaries"][0]["min_by_leading_index"], [1., 0.])
            self.assertEqual(checkpoints[0]["summaries"][0]["nonpositive_count_by_leading_index"], [0, 2])
            self.assertEqual(checkpoints[1]["summaries"][0]["true_count"], 1)
            self.assertEqual(checkpoints[1]["summaries"][0]["false_indices"], [[1]])
            self.assertTrue(checkpoints[2]["first_bad_for_trial"])
            self.assertFalse(checkpoints[3]["first_bad_for_trial"])
            self.assertEqual(sink.first_bad, (1, "root"))
            self.assertIsNone(sink.last)
            self.assertIsNone(sink.anchor)

    def test_raw_nonfinite_jacobian_is_saved_before_existing_zero_replacement(self):
        problem = MixedProblem()
        initial = problem.evaluate(problem.x0)
        initial.jacobian[0, 0] = np.nan

        def solve(fun, x0, *, jac, **kwargs):
            fun(x0)
            expected = np.asarray([[0., 3.], [-4., 5.]])
            np.testing.assert_array_equal(jac(x0), expected)
            return SimpleNamespace(x=x0)

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            with patch("scipy.optimize.least_squares", solve):
                optimization.least_squares(problem, initial_evaluation=initial, diagnostics_dir=directory)
            run, events = records(directory)
            trial = next(e for e in events if e["event"] == "trial")
            self.assertFalse(trial["finite"])
            self.assertEqual(trial["nonfinite_jacobian_indices"], [[0, 0]])
            self.assertTrue(np.isnan(np.load(run / trial["snapshot"])["jacobian"][0, 0]))

    def test_fatal_resource_failure_is_recorded_and_does_not_become_a_penalty(self):
        problem = MixedProblem()

        def fail(_):
            raise MemoryError("host allocation failed")

        problem.evaluate = fail
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            with self.assertRaises(MemoryError):
                optimization.least_squares(problem, diagnostics_dir=directory)
            _, events = records(directory)
            self.assertEqual(events[-1]["event"], "aborted")
            self.assertEqual(events[-2]["event"], "failed_trial")
            self.assertEqual(events[-2]["exception_type"], "MemoryError")
            self.assertFalse(reverse_diagnostics_active())

    def test_recorded_scan_guard_still_raises_and_records_its_boundary(self):
        import jax.numpy as jnp
        from NEOPAX._reverse_ad_initial_er import _raise_if_nonfinite_recorded_scan_tree
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            sink = LeastSquaresDiagnostics(directory, MixedProblem())
            with self.assertRaisesRegex(FloatingPointError, "input fixed-table bars"):
                with sink.evaluation_scope():
                    _raise_if_nonfinite_recorded_scan_tree(
                        {"D13": jnp.asarray([[np.nan, 2.]])}, boundary="input fixed-table bars",
                    )
            self.assertFalse(reverse_diagnostics_active())
            _, events = records(directory)
            event = events[-1]
            self.assertEqual(event["stage"], "recorded_scan/input fixed-table bars")
            self.assertFalse(event["all_finite"])

    def test_combined_examples_enable_diagnostics_and_allow_disabling(self):
        from examples.optimization import (
            optimize_geometry_profiles_from_geometry_result as saved,
            optimize_geometry_profiles_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_simple_least_squares as combined,
            optimize_geometry_warm_start_profiles_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport as warm,
        )
        for example, arguments in ((combined, []), (warm, []), (saved, ["--geometry-result-dir", "/mock"])):
            with self.subTest(example=example.__name__):
                args = example.parser().parse_args(arguments)
                options = combined._least_squares_options(args, object(), (-np.inf, np.inf), [1., 1.])
                self.assertIn("diagnostics_dir", options)
                args = example.parser().parse_args(arguments + ["--no-diagnose-optimization"])
                options = combined._least_squares_options(args, object(), (-np.inf, np.inf), [1., 1.])
                self.assertNotIn("diagnostics_dir", options)

    def test_diagnostics_reject_unsupported_solver_interpretations(self):
        with tempfile.TemporaryDirectory() as directory:
            for kwargs in ({"method": "lm"}, {"loss": "soft_l1"}):
                with self.subTest(kwargs=kwargs):
                    with self.assertRaisesRegex(ValueError, "method='trf'"):
                        optimization.least_squares(MixedProblem(), diagnostics_dir=directory, **kwargs)


if __name__ == "__main__":
    unittest.main()
