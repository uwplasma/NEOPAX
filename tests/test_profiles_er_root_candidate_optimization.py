"""Host-only pilot tests with analytic or mocked workers; no plasma solves."""

import copy
from contextlib import nullcontext, redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from examples.optimization import optimize_profiles_er_root_candidate as pilot


LABELS = [*pilot.PROFILES, "vmec:RBC:1:0", "vmec:ZBS:1:0"]
BASE = [2., 1., 1., 1., 1., 1., 0., 0.]
CONFIG = dict(profiles=dict(zip(pilot.PROFILES, BASE[:6], strict=True)),
              geometry={}, neoclassical={"flux_model": "ntx_scan_runtime"},
              transport_solver={"transport_solver_backend": "radau"})


def endpoint(point, *, root=None, cost=None):
    root = 0.5 + 0.05 * (point[0] - 1.) if root is None else root
    cost = 0.5 * (point[0] - 1.)**2 if cost is None else cost
    rho = np.linspace(0.01, 0.99, 11)
    return dict(parameter_values=list(point), final_time=20., terminal_rho=rho.tolist(),
                terminal_Er=(100 * (root - rho)).tolist(),
                metrics=dict(nearest_downcrossing=root, wrong_sign_fraction=0.,
                             candidates={pilot.CANDIDATE: dict(cost=cost)}))


def report(point=BASE):
    result = endpoint(point)
    gradient = dict.fromkeys(LABELS, 0.)
    gradient.update(n0=point[0] - 1.)
    gradient["vmec:RBC:1:0"] = 1e6  # This must not move geometry or set the profile step norm.
    return dict(all_finite=True, accepted_step_limit=None, parameter_order=LABELS,
                reverse_replay_jacobian_reuse_mode="recorded_forward", reverse_segment_length=50,
                root_loss_audit=dict(
                    target=0.5, nearest_downcrossing=result["metrics"]["nearest_downcrossing"],
                    coordinate_scales=[1.] * len(LABELS), terminal_Er=result["terminal_Er"],
                    terminal_rho=result["terminal_rho"], terminal_time=20.,
                    settings=dict(target=.5, inner_radius=.3, outer_radius=.9,
                                  profile_trust_multiplier=1., geometry_ess_alpha=1.2),
                    experiment=dict(resolved_config=copy.deepcopy(CONFIG), parameter_values=list(point),
                                    initial_er_root_ad="jax_selected_root"),
                    comparisons=[dict(name=pilot.CANDIDATE, cost=result["metrics"]["candidates"][pilot.CANDIDATE]["cost"],
                                      physical_parameter_gradient=gradient,
                                      groups={"profiles": dict(scaled_gradient_l2=abs(point[0]-1.),
                                          physical_descent_direction=[-1., 0., 0., 0., 0., 0., 0., 0.])})]))


class AnalyticEvaluator:
    def __init__(self):
        self.forward_points, self.ad_points = [], []

    def forward(self, point):
        self.forward_points.append(list(point))
        return endpoint(point)

    def ad(self, point):
        self.ad_points.append(list(point))
        return report(point)


class ProfileRootOptimizationTests(unittest.TestCase):
    def test_accepted_moves_recompute_gradients_and_preserve_geometry(self):
        evaluator, events = AnalyticEvaluator(), []
        result = pilot.optimize(report(), evaluator,
                                pilot.PilotOptions(max_iterations=2, initial_step=.1, max_step=.5), emit=events.append)
        self.assertEqual(result["status"], "iteration_limit")
        self.assertFalse(result["converged"])
        self.assertEqual(len(evaluator.ad_points), 2)
        np.testing.assert_allclose([p[0] for p in evaluator.ad_points], [1.9, 1.75])
        for point in evaluator.forward_points + evaluator.ad_points:
            np.testing.assert_array_equal(point[1:], BASE[1:])
        self.assertEqual([e["event"] for e in events],
                         ["gradient_started", "accepted", "gradient_started", "accepted"])
        # Cost stays the scalar 0.5*(n0-1)^2; the pilot does not square it again.
        self.assertAlmostEqual(result["history"][-1]["cost"], .5 * .75**2)

    def test_loss_increasing_branch_trial_backtracks_without_computing_its_ad(self):
        evaluator, events = AnalyticEvaluator(), []
        ordinary_forward = evaluator.forward

        def with_jump(point):
            result = ordinary_forward(point)
            if point[0] < 1.8:
                result["metrics"]["candidates"][pilot.CANDIDATE]["cost"] = 100.
            return result

        evaluator.forward = with_jump
        result = pilot.optimize(report(), evaluator,
                                pilot.PilotOptions(max_iterations=1, initial_step=.5, max_step=.5), emit=events.append)
        np.testing.assert_allclose([p[0] for p in evaluator.forward_points], [2., 1.5, 1.75, 1.875])
        self.assertEqual(len(evaluator.ad_points), 1)
        self.assertEqual(result["history"][-1]["step"], .125)
        self.assertEqual(sum(e["event"] == "rejected" for e in events), 2)

    def test_nonfinite_reverse_trial_does_not_replace_last_accepted_state(self):
        evaluator, events = AnalyticEvaluator(), []
        ordinary_ad = evaluator.ad

        def sometimes_nonfinite(point):
            value = ordinary_ad(point)
            value["all_finite"] = point[0] > 1.92
            return value

        evaluator.ad = sometimes_nonfinite
        result = pilot.optimize(report(), evaluator,
                                pilot.PilotOptions(max_iterations=1, initial_step=.1, max_step=.1), emit=events.append)
        self.assertAlmostEqual(result["parameters"][0], 1.95)
        self.assertEqual(len(evaluator.ad_points), 2)
        self.assertIn("Invalid accepted-point AD", next(e["reason"] for e in events if e["event"] == "rejected"))

    def test_lost_crossing_and_incomplete_forward_are_rejected_before_ad(self):
        for failure in ("root", "time"):
            with self.subTest(failure=failure):
                evaluator = AnalyticEvaluator()
                ordinary = evaluator.forward

                def bad_forward(point):
                    result = ordinary(point)
                    if point[0] < BASE[0]:
                        if failure == "root":
                            result["metrics"]["nearest_downcrossing"] = None
                        else:
                            result["final_time"] = 1.
                    return result

                evaluator.forward = bad_forward
                result = pilot.optimize(report(), evaluator,
                                        pilot.PilotOptions(max_iterations=1, max_backtracks=2))
                self.assertEqual(result["status"], "backtracking_exhausted")
                self.assertEqual(result["parameters"], BASE)
                self.assertEqual(evaluator.ad_points, [])

    def test_cheaper_cost_with_unmoved_root_is_not_reported_as_convergence(self):
        evaluator = AnalyticEvaluator()
        ordinary_forward, ordinary_ad = evaluator.forward, evaluator.ad

        def fixed_root_forward(point):
            result = ordinary_forward(point)
            frozen = endpoint(point, root=.55)
            result.update(terminal_Er=frozen["terminal_Er"])
            result["metrics"]["nearest_downcrossing"] = .55
            return result

        def fixed_root_ad(point):
            result = ordinary_ad(point)
            result["root_loss_audit"]["terminal_Er"] = endpoint(point, root=.55)["terminal_Er"]
            return result

        evaluator.forward, evaluator.ad = fixed_root_forward, fixed_root_ad
        result = pilot.optimize(report(), evaluator, pilot.PilotOptions(max_iterations=1))
        self.assertLess(result["history"][-1]["cost"], result["history"][0]["cost"])
        self.assertAlmostEqual(result["history"][-1]["root_error"], .05)
        self.assertFalse(result["converged"])

    def test_resume_uses_latest_accepted_gradient_and_does_not_repeat_baseline(self):
        first = pilot.optimize(report(), AnalyticEvaluator(),
                               pilot.PilotOptions(max_iterations=1, initial_step=.1, max_step=.5))
        evaluator = AnalyticEvaluator()
        result = pilot.optimize(report(), evaluator,
                                pilot.PilotOptions(max_iterations=2, initial_step=.1, max_step=.5), state=first)
        self.assertEqual(len(evaluator.forward_points), 1)
        self.assertEqual(len(evaluator.ad_points), 1)
        self.assertAlmostEqual(result["parameters"][0], 1.75)

    def test_forward_cache_requires_identical_physics_and_valid_task_digest(self):
        seed = report()
        base = pilot.validate_seed(seed, pilot.PilotOptions())
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            task = dict(base, name="baseline")
            cached = {**endpoint(BASE), "task_digest": pilot.descent.task_digest(task)}
            (folder / "baseline_task.json").write_text(json.dumps(task))
            (folder / "baseline.json").write_text(json.dumps(cached))
            evaluator = pilot.SubprocessEvaluator(seed, base, folder, folder)
            evaluator._run = Mock(side_effect=AssertionError("Fresh worker should not run"))
            with redirect_stdout(io.StringIO()):
                self.assertEqual(evaluator.forward(BASE), cached)
            altered = {**base, "source_digest": "changed source"}
            other = pilot.SubprocessEvaluator(seed, altered, folder, folder)
            other._run = Mock(return_value="new evaluation")
            self.assertEqual(other.forward(BASE), "new evaluation")
            other._run.assert_called_once()

    def test_ad_worker_preserves_configuration_and_passes_explicit_profile_point(self):
        from NEOPAX import _orchestrator as orchestrator
        seed = report()
        original = copy.deepcopy(seed)
        point = [1.9, *BASE[1:]]
        args = SimpleNamespace(reverse_geometry_parameter="RBC:1:0,ZBS:1:0")
        runtime, state = object(), object()
        evaluate = Mock(return_value={"mock": "report"})
        benchmark = SimpleNamespace(_benchmark_device_context=lambda _: nullcontext(),
            _geometry_context_from_config=Mock(return_value="geometry"),
            _run_realtime_geometry_optimization_api_smoke=evaluate)
        with patch.dict("sys.modules", {"benchmark_transport_reverse_ad_only": benchmark}), \
             patch.object(pilot, "ad_arguments", return_value=args), \
             patch.object(orchestrator, "build_runtime_context", return_value=(runtime, state)) as build:
            pilot.run_ad_task(dict(seed_report=seed, forward_task=dict(parameter_values=point)))
        self.assertEqual(seed, original)
        self.assertEqual(build.call_args.args[0], CONFIG)
        self.assertEqual(evaluate.call_args.kwargs["config"]["profiles"], CONFIG["profiles"])
        self.assertEqual(evaluate.call_args.kwargs["parameter_values_override"], point)
        self.assertFalse(evaluate.call_args.kwargs["write_report"])
        self.assertIs(evaluate.call_args.kwargs["baseline_state"], state)

    def test_existing_parser_and_override_keep_validated_replay_and_fixed_geometry(self):
        args = pilot.ad_arguments(report())
        import benchmark_transport_reverse_ad_only as benchmark
        self.assertEqual(benchmark._validate_replay_jacobian_reuse_mode(args, CONFIG), "recorded_forward")
        self.assertIsNone(args.accepted_step_limit)
        self.assertEqual(args.reverse_segment_length, 50)
        self.assertEqual(args.root_loss_audit_target, .5)
        values = benchmark._root_audit_parameter_point(np.array(BASE), [1.9, *BASE[1:]], enabled=True)
        self.assertEqual(float(values[0]), 1.9)
        np.testing.assert_array_equal(np.asarray(values)[6:], BASE[6:])
        for bad, enabled in (([1.9, *BASE[1:6], .1, 0.], True),
                              ([np.nan, *BASE[1:]], True), ([0., *BASE[1:]], True),
                              ([1.9, *BASE[1:]], False)):
            with self.subTest(point=bad, enabled=enabled), self.assertRaises(ValueError):
                benchmark._root_audit_parameter_point(np.array(BASE), bad, enabled=enabled)

    def test_checkpoint_outputs_report_root_and_status_not_only_loss(self):
        result = pilot.optimize(report(), AnalyticEvaluator(), pilot.PilotOptions(max_iterations=1))
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            pilot.save_outputs(folder, result, "test-signature")
            saved = json.loads((folder / "checkpoint.json").read_text())
            summary = json.loads((folder / "summary.json").read_text())
            self.assertEqual(saved["signature"], "test-signature")
            self.assertEqual(saved["state"], result)
            self.assertFalse(summary["converged"])
            self.assertEqual(summary["accepted_iterations"], 1)
            self.assertIn("root_error", summary["history"][-1])
            self.assertIn("root_error", (folder / "history.csv").read_text())

    def test_root_tolerance_is_a_real_crossing_success_condition(self):
        point = [1., *BASE[1:]]
        evaluator = AnalyticEvaluator()
        result = pilot.optimize(report(point), evaluator, pilot.PilotOptions())
        self.assertEqual(result["status"], "root_target_met")
        self.assertTrue(result["converged"])
        self.assertEqual(evaluator.ad_points, [])

    def test_cli_checkpoint_resume_can_extend_budget_without_any_physics_worker(self):
        evaluators = []

        def make_evaluator(*_args):
            instance = AnalyticEvaluator()
            evaluators.append(instance)
            return instance

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            seed_path = folder / "seed.json"
            seed_path.write_text(json.dumps(report()))
            common = ["--ad-report", str(seed_path), "--out-dir", str(folder / "pilot"),
                      "--initial-step", ".1", "--max-step", ".5"]
            with patch.object(pilot, "SubprocessEvaluator", side_effect=make_evaluator), redirect_stdout(io.StringIO()):
                pilot.main([*common, "--max-iterations", "1"])
                pilot.main([*common, "--max-iterations", "2", "--resume"])
            result = json.loads((folder / "pilot/summary.json").read_text())
            self.assertEqual(result["accepted_iterations"], 2)
            self.assertEqual(len(evaluators[1].forward_points), 1)
            self.assertAlmostEqual(result["parameters"][0], 1.75)


if __name__ == "__main__":
    unittest.main()
