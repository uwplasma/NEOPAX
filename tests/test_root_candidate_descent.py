"""Planning and interpretation checks; never launch forward physics workers."""

import copy
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from examples.benchmarks import check_er_root_candidate_descent as descent


def example_report():
    groups = {}
    for name, vector in (("profiles", [-2., 0.]), ("geometry", [0., 0.3])):
        groups[name] = dict(scaled_gradient_l2=2., physical_descent_direction=vector,
                            predicted_cost_change_per_unit_descent=-2., predicted_position_cost_change=-0.1)
    return dict(all_finite=True, accepted_step_limit=None, parameter_order=["n0", "vmec:RBC:1:0"], root_loss_audit=dict(
        settings={}, target=0.514,
        experiment=dict(resolved_config={}, parameter_values=[4., 0.], initial_er_root_ad="jax_selected_root"),
        comparisons=[dict(name="audit_root_local_cost", groups=groups), dict(name="audit_root_annular_cost", groups=groups)],
    ))


class RootCandidateDescentTests(unittest.TestCase):
    def test_planner_uses_saved_physical_directions_and_preserves_other_group(self):
        report = example_report()
        original = copy.deepcopy(report)
        tasks = descent.plan_checks(report, [0.01])
        self.assertEqual(len(tasks), 5)
        np.testing.assert_array_equal(tasks[0]["parameter_values"], [4., 0.])
        np.testing.assert_array_equal(tasks[1]["parameter_values"], [3.98, 0.])
        np.testing.assert_array_equal(tasks[2]["parameter_values"], [4., 0.003])
        self.assertEqual(report, original)
        for task in tasks:
            self.assertEqual(task["resolved_config"], report["root_loss_audit"]["experiment"]["resolved_config"])
        self.assertNotEqual(descent.task_digest(tasks[0]), descent.task_digest(tasks[1]))

    def test_bad_steps_prefix_reports_and_nonfinite_reports_fail_before_launch(self):
        for step in (0, -0.1, np.nan, np.inf, 3.):
            with self.subTest(step=step), self.assertRaises(ValueError):
                descent.plan_checks(example_report(), [step])
        for change in (dict(all_finite=False), dict(accepted_step_limit=16)):
            with self.assertRaises(ValueError):
                descent.plan_checks({**example_report(), **change}, [0.01])

    def test_interpretation_detects_cheaper_loss_with_weaker_field_and_unmoved_root(self):
        task = descent.plan_checks(example_report(), [0.01])[1]
        name = task["candidate"]
        metrics = dict(candidates={name: dict(cost=10., zero_cost=8., sign_cost=2.)},
                       position_cost=0.02, nearest_downcrossing=0.714, nearest_downcrossing_bracket=35,
                       annular_Er_rms=10.)
        baseline = dict(metrics=metrics)
        weaker = copy.deepcopy(baseline)
        weaker["metrics"]["candidates"][name].update(cost=8., zero_cost=5., sign_cost=3.)
        weaker["metrics"]["annular_Er_rms"] = 8.
        result = descent.compare_endpoint(task, baseline, weaker, example_report()["root_loss_audit"])
        self.assertTrue(result["cost_decreased"])
        self.assertFalse(result["root_position_improved"])
        self.assertTrue(result["cost_decreased_without_position_improvement"])
        self.assertEqual(result["annular_Er_rms_change"], -2.)
        self.assertEqual(result["zero_cost_change"], -3.)
        self.assertEqual(result["sign_cost_change"], 1.)
        self.assertAlmostEqual(result["predicted_cost_change"], -0.02)

    def test_missing_root_is_unavailable_and_creation_is_reported_separately(self):
        task = descent.plan_checks(example_report(), [0.01])[1]
        name = task["candidate"]
        metrics = dict(candidates={name: dict(cost=10., zero_cost=8., sign_cost=2.)},
                       position_cost=None, nearest_downcrossing=None, nearest_downcrossing_bracket=None,
                       annular_Er_rms=10.)
        baseline = dict(metrics=metrics)
        created = copy.deepcopy(baseline)
        created["metrics"].update(position_cost=0.001, nearest_downcrossing=0.55, nearest_downcrossing_bracket=27)
        result = descent.compare_endpoint(task, baseline, created, example_report()["root_loss_audit"])
        self.assertIsNone(result["root_position_improved"])
        self.assertTrue(result["root_created"])
        self.assertFalse(result["root_lost"])

    def test_forward_worker_rebuilds_changed_profiles_and_geometry_then_runs_full_adaptive_solve(self):
        from NEOPAX import _geometry_autodiff as geometry, _orchestrator as orchestrator, _reverse_ad_transport as reverse
        from NEOPAX._reverse_ad_parameters import PROFILE_PARAMETER_ORDER

        @dataclass
        class State:
            Er: object

        rho = (np.arange(51) + 0.5) / 51
        state = State(500 * (0.55 - rho))
        runtime = SimpleNamespace(geometry=SimpleNamespace(rho_grid=rho))
        labels = [*PROFILE_PARAMETER_ORDER, "vmec:RBC:1:0", "vmec:ZBS:1:0"]
        values = [4.2, 17.8, 10., 2., 1., 1., 0.001, -0.002]
        task = dict(name="probe", parameter_labels=labels, parameter_values=values,
                    resolved_config={"profiles": {}, "geometry": {}}, settings={}, initial_er_root_ad="jax_selected_root")
        build_runtime = Mock(return_value=(runtime, state))
        build_state = Mock(return_value=state)
        input_object = SimpleNamespace(to_indata=lambda path: Path(path).write_text("mock input"))
        apply_deltas = Mock(return_value=input_object)
        rollout = SimpleNamespace(final_carry=SimpleNamespace(t=20.), accepted_count=290, attempt_count=495)
        schedule = Mock(return_value=(None, None, None, rollout, state))
        siblings = {
            "benchmark_transport_reverse_ad_only": SimpleNamespace(
                _geometry_context_from_config=lambda *_: object(), _benchmark_device_context=lambda _: nullcontext()),
            "benchmark_transport_realtime_geometry_forward_fd": SimpleNamespace(_schedule_rollout=schedule),
        }
        with patch.dict("sys.modules", siblings), patch.object(orchestrator, "build_runtime_context", build_runtime), \
             patch.object(reverse, "initial_state_for_parameter_vector", build_state), \
             patch.object(geometry, "boundary_param_entries", lambda *_: object()), \
             patch.object(geometry, "_input_with_boundary_deltas", apply_deltas):
            result = descent.run_forward_task(task)
        cfg = build_runtime.call_args.args[0]
        self.assertEqual(cfg["profiles"]["density_shape_power"], 10.)
        self.assertEqual(cfg["geometry"]["vmec_param_delta"], 0.)
        self.assertNotIn("vmec_input_file", task["resolved_config"]["geometry"])
        np.testing.assert_array_equal(apply_deltas.call_args.args[1], [0.001, -0.002])
        self.assertIsNone(schedule.call_args.kwargs["accepted_step_limit"])
        self.assertEqual(build_state.call_args.kwargs["initial_er_root_ad"], "off")
        np.testing.assert_array_equal(schedule.call_args.args[2].Er, state.Er)
        self.assertEqual(result["final_time"], 20.)
        self.assertEqual(result["accepted_steps"], 290)
        self.assertAlmostEqual(result["metrics"]["nearest_downcrossing"], 0.55)


if __name__ == "__main__":
    unittest.main()
