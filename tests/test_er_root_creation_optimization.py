"""Analytic fields and mocked workers only: never execute plasma physics."""

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

from examples.optimization import optimize_er_root_candidates as creation
from examples.optimization import optimize_profiles_er_root_candidate as pilot


LABELS = [*pilot.PROFILES, "vmec:RBC:1:0", "vmec:ZBS:1:0"]
BASE = [2., 1., 1., 1., 1., 1., 0., 0.]
SETTINGS = dict(target=.5, inner_radius=.3, outer_radius=.9,
                profile_trust_multiplier=1., geometry_ess_alpha=1.2)


def endpoint(point):
    control = 2 - point[0] + point[6]
    crossing = .1 + .5 * control
    rho = np.linspace(.05, .95, 19)
    er = crossing - rho
    costs = {pilot.cost_name(name): dict(cost=.5 * (control - goal)**2)
             for name, goal in (("local", 1.), ("annular", .8))}
    return dict(parameter_values=list(point), final_time=2., terminal_rho=rho.tolist(), terminal_Er=er.tolist(),
                metrics=dict(nearest_downcrossing=crossing if .3 <= crossing <= .9 else None,
                             wrong_sign_fraction=1., candidates=costs))


def report(point=BASE):
    result = endpoint(point)
    control = 2 - point[0] + point[6]
    comparisons = []
    for name, goal in (("local", 1.), ("annular", .8)):
        gradient = dict.fromkeys(LABELS, 0.)
        gradient.update(n0=goal - control)
        gradient[LABELS[6]] = control - goal
        comparisons.append(dict(name=pilot.cost_name(name), cost=.5 * (control-goal)**2,
            physical_parameter_gradient=gradient,
            groups={"profiles": dict(scaled_gradient_l2=abs(control-goal),
                                     physical_descent_direction=[-1., 0., 0., 0., 0., 0., 0., 0.]),
                    "geometry": dict(scaled_gradient_l2=abs(control-goal),
                                     physical_descent_direction=[0., 0., 0., 0., 0., 0., 1., 0.])}))
    return dict(all_finite=True, accepted_step_limit=None, parameter_order=LABELS,
        reverse_replay_jacobian_reuse_mode="recorded_forward", reverse_segment_length=50,
        baseline_no_electron_region_verified=True,
        optimization_start_case=dict(seed_input="original.input", mboz=18, nboz=18,
                                     surfaces=[.02, .1, .5, 1.], max_reverse_accepted_steps=500),
        root_loss_audit=dict(target=.5, nearest_downcrossing=result["metrics"]["nearest_downcrossing"],
            coordinate_scales=[1.] * len(LABELS), terminal_Er=result["terminal_Er"],
            terminal_rho=result["terminal_rho"], terminal_time=2., settings=SETTINGS,
            experiment=dict(resolved_config=dict(profiles=dict(zip(pilot.PROFILES, BASE[:6], strict=True)),
                geometry={}, transport_solver={}), parameter_values=list(point), initial_er_root_ad="jax_selected_root"),
            comparisons=comparisons))


class AnalyticEvaluator:
    def __init__(self):
        self.forward_points, self.ad_points = [], []

    def forward(self, point):
        self.forward_points.append(list(point))
        return endpoint(point)

    def ad(self, point):
        self.ad_points.append(list(point))
        return report(point)


class RootCreationOptimizationTests(unittest.TestCase):
    def test_verified_saved_geometry_may_already_have_a_root(self):
        seed = report([1., *BASE[1:]])
        seed["optimization_start_case"].update(case_kind="saved_geometry_result")
        seed.update(baseline_case_verified=True, baseline_no_electron_region_verified=False)
        creation.validate_creation_seed(seed)
        seed["baseline_case_verified"] = False
        with self.assertRaises(ValueError):
            creation.validate_creation_seed(seed)

    def test_initial_crossing_is_not_counted_as_new_root_creation(self):
        seed = report([1., *BASE[1:]])
        state = pilot.optimize(seed, AnalyticEvaluator(), pilot.PilotOptions(allow_root_creation=True, max_iterations=1))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "annular_profiles"
            run.mkdir()
            pilot.save_outputs(run, state, "test")
            rows = creation.save_comparison(root, "seed")
            self.assertTrue(rows[0]["initial_has_downcrossing"])
            self.assertFalse(rows[0]["downcrossing_created"])
            self.assertAlmostEqual(rows[0]["initial_root"], .6)

    def test_geometry_baseline_ad_is_prepared_once_and_rejects_changed_inputs(self):
        from examples.benchmarks import benchmark_er_root_candidates_optimization_start as baseline
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            geometry, out = root / "geometry", root / "comparison"
            geometry.mkdir()
            boundary = geometry / "input.QI_neopax_database_full_transport_net_power_optimized"
            boundary.write_text("accepted boundary")
            (geometry / "optimization_summary.json").write_text(json.dumps(dict(
                parameter_labels=["RBC:1:0"], max_mode_schedule=[2])))
            def prepare(command, **kwargs):
                _, _, source = baseline.load_saved_geometry(geometry)
                saved = report()
                saved["optimization_start_case"].update(case_kind="saved_geometry_result", saved_geometry_result=source)
                path = out / "baseline" / "ad_summary.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(saved))
                self.assertIn("--ad-only", command)
                self.assertIn("--geometry-result-dir", command)
                return SimpleNamespace(returncode=0)
            with patch.object(creation.subprocess, "run", side_effect=prepare) as start:
                first = creation.geometry_result_ad_report(geometry, out)
                self.assertEqual(first, creation.geometry_result_ad_report(geometry, out))
                start.assert_called_once()
                boundary.write_text("changed boundary")
                with self.assertRaisesRegex(ValueError, "differs"):
                    creation.geometry_result_ad_report(geometry, out)
                start.assert_called_once()

    def test_root_free_seed_and_trials_are_allowed_explicitly(self):
        seed = report()
        with patch.object(pilot.descent, "source_digest", return_value="physics"):
            with self.assertRaisesRegex(ValueError, "existing annular"):
                pilot.validate_seed(seed, pilot.PilotOptions())
            for name in ("local", "annular"):
                options = pilot.PilotOptions(candidate=name, allow_root_creation=True,
                    initial_step=.2, max_step=.4, growth_factor=2., max_iterations=2)
                pilot.validate_seed(seed, options)
                evaluator = AnalyticEvaluator()
                state = pilot.optimize(seed, evaluator, options)
                self.assertIsNone(state["history"][0]["root"])
                self.assertIsNone(state["history"][1]["root"])
                self.assertTrue(state["history"][2]["has_downcrossing"])
                self.assertEqual(len(evaluator.ad_points), 2)
                self.assertGreater(state["history"][2]["inner_positive_fraction"], 0)
                self.assertEqual(state["status"], "iteration_limit")
                self.assertFalse(state["converged"])

    def test_geometry_trial_moves_no_profile_parameter(self):
        evaluator = AnalyticEvaluator()
        state = pilot.optimize(report(), evaluator, pilot.PilotOptions(group="geometry", candidate="local",
            allow_root_creation=True, max_iterations=2, initial_step=.2, max_step=.4, growth_factor=2.))
        self.assertTrue(state["history"][-1]["has_downcrossing"])
        for point in evaluator.forward_points + evaluator.ad_points:
            np.testing.assert_array_equal(point[:6], BASE[:6])
            self.assertEqual(point[7], 0.)

    def test_nonfinite_cost_is_rejected_with_json_serializable_diagnostics(self):
        class BadCostEvaluator(AnalyticEvaluator):
            def forward(self, point):
                result = super().forward(point)
                if point[0] < BASE[0]:
                    result["metrics"]["candidates"][pilot.CANDIDATE]["cost"] = float("nan")
                return result
        evaluator, events = BadCostEvaluator(), []
        def emit(event):
            json.dumps(event, allow_nan=False)
            events.append(event)
        state = pilot.optimize(report(), evaluator, pilot.PilotOptions(
            allow_root_creation=True, max_iterations=1, max_backtracks=2), emit=emit)
        self.assertEqual(state["status"], "backtracking_exhausted")
        self.assertEqual(evaluator.ad_points, [])
        self.assertEqual(len(events), 2)

    def test_completed_own_forward_worker_is_reused_after_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with patch.object(pilot.descent, "source_digest", return_value="physics"):
                baseline = pilot.validate_seed(report(), pilot.PilotOptions(allow_root_creation=True))
            task = dict(baseline, name="completed", parameter_values=[1.9, *BASE[1:]])
            cached = {**endpoint(task["parameter_values"]), "task_digest": pilot.descent.task_digest(task)}
            (folder / "evaluation_0001_forward_task.json").write_text(json.dumps(dict(kind="forward", forward_task=task)))
            (folder / "evaluation_0001_forward.json").write_text(json.dumps(cached))
            evaluator = pilot.SubprocessEvaluator(report(), baseline, folder)
            evaluator._run = Mock(side_effect=AssertionError("Should reuse the completed forward"))
            with redirect_stdout(io.StringIO()):
                self.assertEqual(evaluator.forward(task["parameter_values"]), cached)

    def test_candidate_selection_changes_the_actual_stationary_point(self):
        states = {}
        for name in ("local", "annular"):
            states[name] = pilot.optimize(report(), AnalyticEvaluator(), pilot.PilotOptions(candidate=name,
                allow_root_creation=True, initial_step=.2, max_step=.4, growth_factor=2., max_iterations=5))
        self.assertEqual(states["local"]["status"], "stationary_before_target")
        self.assertFalse(states["local"]["converged"])
        self.assertAlmostEqual(states["local"]["history"][-1]["root"], .6)
        self.assertTrue(states["annular"]["converged"])
        self.assertAlmostEqual(states["annular"]["history"][-1]["root"], .5)

    def test_zero_or_positive_field_with_cheap_cost_is_not_root_success(self):
        for value in (0., .2):
            class NoCrossingEvaluator(AnalyticEvaluator):
                def forward(self, point):
                    result = super().forward(point)
                    if point[0] < BASE[0]:
                        result["terminal_Er"] = [value] * len(result["terminal_Er"])
                    return result

                def ad(self, point):
                    result = super().ad(point)
                    result["root_loss_audit"]["terminal_Er"] = [value] * len(result["root_loss_audit"]["terminal_Er"])
                    return result
            state = pilot.optimize(report(), NoCrossingEvaluator(), pilot.PilotOptions(
                allow_root_creation=True, max_iterations=1))
            self.assertFalse(state["converged"])
            self.assertFalse(state["history"][-1]["has_downcrossing"])
            self.assertIsNone(state["history"][-1]["root_error"])

    def test_unverified_existing_root_or_all_positive_seed_is_rejected(self):
        for seed in (report([1., *BASE[1:]]), {**report(), "baseline_no_electron_region_verified": False}):
            with self.assertRaises(ValueError):
                creation.validate_creation_seed(seed)
        seed = report()
        seed["root_loss_audit"]["terminal_Er"] = [1.] * 19
        with self.assertRaises(ValueError):
            creation.validate_creation_seed(seed)

    def test_geometry_ad_points_are_explicitly_validated(self):
        from examples.benchmarks import benchmark_transport_reverse_ad_only as benchmark
        point = [*BASE[:6], .01, 0.]
        with self.assertRaises(ValueError):
            benchmark._root_audit_parameter_point(np.asarray(BASE), point, enabled=True)
        actual = benchmark._root_audit_parameter_point(np.asarray(BASE), point, enabled=True, allow_geometry=True)
        np.testing.assert_array_equal(np.asarray(actual), point)

    def test_ad_worker_reuses_original_geometry_context_and_creation_rows(self):
        from NEOPAX import _orchestrator, optimization
        from examples.benchmarks._er_root_loss_audit import ROOT_CREATION_OBJECTIVES
        seed = report()
        original = copy.deepcopy(seed)
        args = SimpleNamespace(reverse_geometry_parameter="RBC:1:0,ZBS:1:0")
        evaluate = Mock(return_value={"all_finite": True})
        benchmark = SimpleNamespace(_benchmark_device_context=lambda _: nullcontext(),
            _geometry_context_from_config=Mock(side_effect=AssertionError("Must preserve seed context")),
            _run_realtime_geometry_optimization_api_smoke=evaluate)
        point = [*BASE[:6], .01, 0.]
        with patch.dict("sys.modules", {"benchmark_transport_reverse_ad_only": benchmark}), \
             patch.object(pilot, "ad_arguments", return_value=args), \
             patch.object(_orchestrator, "build_runtime_context", return_value=("runtime", "state")), \
             patch.object(optimization, "build_geometry_autodiff_context", return_value="original context") as context:
            result = pilot.run_ad_task(dict(seed_report=seed, forward_task=dict(parameter_values=point)))
        self.assertEqual(seed, original)
        self.assertEqual(context.call_args.kwargs["mboz"], 18)
        self.assertEqual(context.call_args.kwargs["surface_s"], (.02, .1, .5, 1.))
        self.assertEqual(args.max_reverse_accepted_steps, 500)
        self.assertTrue(args.root_loss_audit_allow_geometry)
        self.assertEqual(evaluate.call_args.kwargs["root_audit_objective_labels"], ROOT_CREATION_OBJECTIVES)
        self.assertEqual(evaluate.call_args.kwargs["parameter_values_override"], point)
        self.assertEqual(result["optimization_start_case"], seed["optimization_start_case"])

    def test_comparison_cli_uses_two_independent_seeds_and_can_resume(self):
        evaluators = []
        def make_evaluator(*_args):
            evaluator = AnalyticEvaluator()
            evaluators.append(evaluator)
            return evaluator
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            seed_path, out = folder / "seed.json", folder / "comparison"
            seed_path.write_text(json.dumps(report()))
            common = ["--ad-report", str(seed_path), "--out-dir", str(out), "--initial-step", ".2", "--max-step", ".4"]
            with patch.object(pilot, "SubprocessEvaluator", side_effect=make_evaluator), \
                 patch.object(pilot.descent, "source_digest", return_value="physics"), \
                 patch.object(creation, "plot_comparison"), redirect_stdout(io.StringIO()):
                creation.main([*common, "--max-iterations", "1"])
                creation.main([*common, "--max-iterations", "2", "--resume"])
            self.assertEqual(evaluators[0].forward_points[0], BASE)
            self.assertEqual(evaluators[1].forward_points[0], BASE)
            self.assertEqual(len(evaluators[2].forward_points), 1)
            rows = json.loads((out / "comparison.json").read_text())["runs"]
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertEqual(row["first_downcrossing_iteration"], 2)
                self.assertFalse(row["converged"])
                self.assertTrue(row["final_has_downcrossing"])
            self.assertIn("final_inner_positive_fraction", (out / "comparison.csv").read_text())


if __name__ == "__main__":
    unittest.main()
