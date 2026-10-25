"""Planning and interpretation checks; never launch forward physics workers."""

import copy
from contextlib import nullcontext
from contextlib import redirect_stdout
from dataclasses import dataclass
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import tempfile

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

    def test_root_creation_check_has_no_position_prediction_but_reports_inner_field_progress(self):
        task = descent.plan_checks(example_report(), [.001])[1]
        task["prediction"]["predicted_position_cost_change"] = None
        name = task["candidate"]
        metrics = dict(candidates={name: dict(cost=10., zero_cost=5., sign_cost=5.)},
                       position_cost=None, nearest_downcrossing=None, nearest_downcrossing_bracket=None,
                       annular_Er_rms=10., inner_mean_Er=-10., inner_positive_fraction=0., wrong_sign_fraction=.5)
        baseline = dict(metrics=metrics)
        endpoint = copy.deepcopy(baseline)
        endpoint["metrics"].update(inner_mean_Er=-8., inner_positive_fraction=.1, wrong_sign_fraction=.4)
        result = descent.compare_endpoint(task, baseline, endpoint, example_report()["root_loss_audit"])
        self.assertIsNone(result["predicted_position_cost_change"])
        self.assertIsNone(result["root_position_improved"])
        self.assertFalse(result["root_created"])
        self.assertEqual(result["inner_mean_Er_change"], 2.)
        self.assertAlmostEqual(result["inner_positive_fraction_change"], .1)
        self.assertAlmostEqual(result["wrong_sign_fraction_change"], -.1)

    def test_forward_worker_matches_fixed_config_profile_map_and_requested_root_initializer(self):
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
                    resolved_config={"profiles": {"density_shape_power": 9., "T0": 17.7}, "geometry": {}},
                    settings={}, initial_er_root_ad="jax_selected_root")
        build_runtime = Mock(return_value=(runtime, state))
        selected_state = State(500 * (0.551 - rho))
        build_state = Mock(return_value=selected_state)
        input_object = SimpleNamespace(to_indata=lambda path: Path(path).write_text("mock input"))
        apply_deltas = Mock(return_value=input_object)
        rollout = SimpleNamespace(final_carry=SimpleNamespace(t=20.), accepted_count=290, attempt_count=495)
        schedule = Mock(return_value=(None, None, None, rollout, state))
        siblings = {
            "benchmark_transport_reverse_ad_only": SimpleNamespace(
                _geometry_context_from_config=lambda *_: object(), _benchmark_device_context=lambda _: nullcontext()),
            "benchmark_transport_realtime_geometry_forward_fd": SimpleNamespace(_schedule_rollout=schedule),
        }
        siblings["benchmark_transport_realtime_geometry_forward_fd"]._adaptive_rollout_diagnostics = lambda _: {"accepted_count": 290}
        with patch.dict("sys.modules", siblings), patch.object(orchestrator, "build_runtime_context", build_runtime), \
             patch.object(reverse, "initial_state_for_parameter_vector", build_state), \
             patch.object(geometry, "boundary_param_entries", lambda *_: object()), \
             patch.object(geometry, "_input_with_boundary_deltas", apply_deltas):
            result = descent.run_forward_task(task)
        cfg = build_runtime.call_args.args[0]
        self.assertEqual(cfg["profiles"]["density_shape_power"], 9.)
        self.assertEqual(cfg["profiles"]["T0"], 17.7)
        np.testing.assert_array_equal(build_state.call_args.args[0], values[:6])
        self.assertEqual(cfg["geometry"]["vmec_param_delta"], 0.)
        self.assertNotIn("vmec_input_file", task["resolved_config"]["geometry"])
        np.testing.assert_array_equal(apply_deltas.call_args.args[1], [0.001, -0.002])
        self.assertIsNone(schedule.call_args.kwargs["accepted_step_limit"])
        self.assertEqual(build_state.call_args.kwargs["initial_er_root_ad"], "jax_selected_root")
        np.testing.assert_array_equal(schedule.call_args.args[2].Er, selected_state.Er)
        np.testing.assert_array_equal(result["initial_state"]["Er"], selected_state.Er)
        np.testing.assert_array_equal(result["runtime_initial_Er"], state.Er)
        self.assertEqual(result["final_time"], 20.)
        self.assertEqual(result["accepted_steps"], 290)
        self.assertAlmostEqual(result["metrics"]["nearest_downcrossing"], 0.55)

    def test_initial_root_diagnostics_distinguish_initial_sign_changes_from_final_bracket_changes(self):
        task = descent.plan_checks(example_report(), [0.01])[1]
        name = task["candidate"]
        metrics = dict(candidates={name: dict(cost=10., zero_cost=8., sign_cost=2.)},
                       position_cost=0.02, nearest_downcrossing=0.714, nearest_downcrossing_bracket=35,
                       annular_Er_rms=10.)
        baseline = dict(metrics=metrics, initial_state={"Er": [0., 10., -10.]})
        endpoint = copy.deepcopy(baseline)
        endpoint["initial_state"]["Er"] = [0., -5., -9.]
        result = descent.compare_endpoint(task, baseline, endpoint, example_report()["root_loss_audit"])
        self.assertFalse(result["root_bracket_changed"])
        self.assertEqual(result["initial_root_changes"]["sign_change_indices"], [1])
        self.assertEqual(result["initial_root_changes"]["max_abs_change"], 15.)

    def test_initial_root_comparison_separates_smooth_continuation_and_entrypoint_disagreement(self):
        rho = np.array([0.1, 0.4, 0.6, 0.9])
        initial = np.array([0., 3., -4., -5.])
        rate = np.array([np.nan, -2., 1., 0.5])
        step = 1e-3
        selected = initial + step * np.nan_to_num(rate)
        smooth = descent.compare_initial_roots(initial, selected, selected, rate, step, rho, [0.3, 0.9])
        self.assertLess(smooth["annular_rate_error_l2_relative"], 1e-12)
        self.assertEqual(smooth["selected_initial_sign_change_indices"], [])
        self.assertEqual(smooth["selected_minus_general_max_abs"], 0.)
        switched = selected.copy()
        switched[1] = -6.
        different = descent.compare_initial_roots(initial, switched, selected, rate, step, rho, [0.3, 0.9])
        self.assertEqual(different["selected_initial_sign_change_indices"], [1])
        self.assertGreater(different["selected_minus_general_max_abs"], 8.)
        self.assertGreater(different["annular_rate_error_l2_relative"], 0.9)
        self.assertIsNone(different["predicted_initial_Er_change"][0])

    def test_actual_profile_builder_shows_config_mutation_adds_frozen_helium_temperature_dependence(self):
        import jax.numpy as jnp
        from NEOPAX._orchestrator import _build_state
        from NEOPAX._profiles import build_profiles
        from NEOPAX._reverse_ad_transport import initial_state_for_parameter_vector

        field = SimpleNamespace(r_grid=jnp.linspace(0.1, 0.9, 5), r_grid_half=jnp.linspace(0., 1., 6))
        species = SimpleNamespace(number_species=4, charge_qp=jnp.array([-1., 1., 1., 2.]))
        profiles = dict(n0=4.21, n_edge=0.6, T0=17.8, T_edge=0.7, c_density=[1., .5, .5, 0.],
                        c_temperature=[1.] * 4, density_shape_power=10., temperature_shape_power=2.,
                        density_shape_alpha=1., temperature_shape_alpha=1., T_scale=.7)
        cfg = dict(profiles=profiles, transport_solver={"density_floor": 1e-6})
        baseline = _build_state(cfg, field, species)
        point = jnp.array([4.21, 17.813246655739743, 10., 2., 1., 1.])
        initial = initial_state_for_parameter_vector(
            point, baseline_state=baseline, runtime=SimpleNamespace(geometry=field, species=species),
            config=cfg, profile_cfg=profiles, initial_er_root_ad="off",
        )
        fixed_baseline = build_profiles(profiles, field, 4).temperature[3] / 1e3
        fixed_legacy = build_profiles({**profiles, "T0": float(point[1])}, field, 4).temperature[3] / 1e3
        np.testing.assert_allclose(initial.temperature[3], fixed_legacy, rtol=1e-12)
        self.assertGreater(float(jnp.max(abs(fixed_legacy - fixed_baseline))), 0.005)
        self.assertEqual(cfg["profiles"]["T0"], 17.8)

    def test_initialization_diagnostic_builds_one_runtime_and_never_runs_transport(self):
        from NEOPAX import _orchestrator as orchestrator, _reverse_ad_transport as reverse, _profiles as profiles
        from NEOPAX import _reverse_ad_initial_er as roots
        from NEOPAX._reverse_ad_parameters import PROFILE_PARAMETER_ORDER

        @dataclass
        class State:
            Er: object
            parameters: object

        point = np.array([4., 17.8, 10., 2., 1., 1.])
        rho = np.array([0.1, 0.4, 0.6, 0.9])
        root = lambda p: 0.5 * p[0] + np.array([1., 0., -5., -6.])
        state = State(root(point), point)
        runtime = SimpleNamespace(geometry=SimpleNamespace(rho_grid=rho), species=SimpleNamespace(number_species=4))
        cfg = {"profiles": dict(zip(PROFILE_PARAMETER_ORDER, point, strict=True)),
               "equations": {"toggle_temperature": [True, True, True, False]}}
        report = example_report()
        report["parameter_order"] = [*PROFILE_PARAMETER_ORDER, "vmec:RBC:1:0", "vmec:ZBS:1:0"]
        audit = report["root_loss_audit"]
        audit["annulus"] = [.3, .9]
        audit["experiment"].update(resolved_config=cfg, parameter_values=[*point, 0., 0.])
        for item in audit["comparisons"]:
            item["groups"]["profiles"]["physical_descent_direction"] = [-2., 0., 0., 0., 0., 0., 0., 0.]
        build_runtime = Mock(return_value=(runtime, state))
        def initial(values, **kwargs):
            values = np.asarray(values)
            return State(root(values) if kwargs["initial_er_root_ad"] == "jax_selected_root" else state.Er, values)
        general = Mock(side_effect=lambda _cfg, _runtime, pre: State(root(pre.parameters), pre.parameters))
        schedules = Mock(side_effect=AssertionError("initial-only diagnostic ran transport"))
        siblings = {
            "benchmark_transport_reverse_ad_only": SimpleNamespace(_benchmark_device_context=lambda _: nullcontext()),
            "benchmark_transport_realtime_geometry_forward_fd": SimpleNamespace(_schedule_rollout=schedules),
        }
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", siblings), \
             patch.object(orchestrator, "build_runtime_context", build_runtime), \
             patch.object(orchestrator, "_maybe_initialize_er_from_ambipolarity", general), \
             patch.object(reverse, "initial_state_for_parameter_vector", initial), \
             patch.object(profiles, "build_profiles", lambda *_: SimpleNamespace(temperature=np.ones((4, 4)) * 1000)), \
             patch.object(roots, "initial_er_charge_flux_residuals", lambda pre, er, **_: np.asarray(er) - root(pre.parameters)), \
             patch.object(roots, "initial_er_charge_flux_residual_er_derivative", lambda *_args, **_kwargs: np.ones(4)), \
             redirect_stdout(io.StringIO()):
            result = descent.diagnose_profile_initialization(report, [.001], "annular", Path(directory))
            saved = json.loads((Path(directory) / "initialization_diagnostic.json").read_text())
        self.assertTrue(saved["no_transport_solve"])
        self.assertEqual(build_runtime.call_count, 1)
        schedules.assert_not_called()
        self.assertEqual(general.call_count, 1)
        np.testing.assert_allclose(result["implicit_initial_Er_rate"], -np.ones(4), rtol=1e-8)
        self.assertLess(result["endpoints"][0]["annular_rate_error_l2_relative"], 1e-8)
        self.assertEqual(result["endpoints"][0]["selected_minus_general_max_abs"], 0.)


if __name__ == "__main__":
    unittest.main()
