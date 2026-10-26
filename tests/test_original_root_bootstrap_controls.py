"""Small objective derivatives and mocked optimizers; no physical solves."""

import copy
from contextlib import redirect_stdout, nullcontext
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch, Mock

import jax
import jax.numpy as jnp
import numpy as np

from NEOPAX import _reverse_ad_transport as reverse
from examples.benchmarks import _er_root_loss_audit as audit
from examples.benchmarks import analyze_er_root_saved_profiles as saved
from examples.optimization import optimize_er_root_candidates as comparison
from examples.optimization import optimize_profiles_er_root_candidate as pilot
from tests import test_er_root_creation_optimization as analytic


def legacy_endpoint(point):
    result = analytic.endpoint(point)
    control = 2-point[0]+point[6]
    for name, goal in (("original", 1.), ("bootstrap", .6), ("original_bootstrap", .9)):
        cost = .5*(control-goal)**2
        components = dict(position_cost=cost, strength_cost=0.) if name == "original" else dict(bootstrap_cost=cost)
        result["metrics"]["candidates"][pilot.cost_name(name)] = dict(cost=cost, **components)
    result["metrics"]["original_objectives"] = dict(Er_transition_location_moment=.1+.5*control-.5,
        Er_transition_strength=2., strength_target=.9, strength_deficit=0.,
        bootstrap_current_softmax_abs_scaled=.1+abs(control-.6), bootstrap_limit=.1,
        bootstrap_penalty=abs(control-.6))
    return result


def legacy_report(point=analytic.BASE):
    seed = analytic.report(point)
    endpoint = legacy_endpoint(point)
    seed["optimization_start_case"]["objective_suite"] = "original"
    seed["root_loss_audit"]["original_objective_settings"] = audit.original_objective_settings()
    control = 2-point[0]+point[6]
    for name, goal in (("original", 1.), ("bootstrap", .6), ("original_bootstrap", .9)):
        gradient = dict.fromkeys(analytic.LABELS, 0.)
        gradient.update(n0=goal-control)
        gradient[analytic.LABELS[6]] = control-goal
        directions = {group:dict(scaled_gradient_l2=abs(control-goal),
            physical_descent_direction=[-1., 0., 0., 0., 0., 0., 0., 0.] if group == "profiles"
                                       else [0., 0., 0., 0., 0., 0., 1., 0.])
            for group in ("profiles", "geometry")}
        seed["root_loss_audit"]["comparisons"].append(dict(name=pilot.cost_name(name),
            cost=endpoint["metrics"]["candidates"][pilot.cost_name(name)]["cost"],
            physical_parameter_gradient=gradient, groups=directions))
    return seed


class LegacyEvaluator:
    def __init__(self):
        self.points = []

    def forward(self, point):
        self.points.append(list(point))
        return legacy_endpoint(point)

    def ad(self, point):
        return legacy_report(point)


class OriginalRootBootstrapControlsTests(unittest.TestCase):
    def test_original_weighted_cost_and_chain_rule_match_independent_finite_differences(self):
        options = audit.original_objective_settings()
        self.assertEqual([options[k] for k in ("position_weight", "strength_weight", "bootstrap_weight")], [1000., 100., 4.])
        raw = np.array([.03, .2, .5])
        matrix = np.array([[.001, -.002], [.1, -.2], [.3, .2]])

        def independent(point):
            m, s, b = raw+matrix@point
            deficit = .5*(.9-s+np.hypot(.9-s, 1e-6))
            root = .5*1000*m*m+.5*100*deficit*deficit
            bootstrap = .5*4*max(b-.1, 0.)**2
            return np.array([root, bootstrap, root+bootstrap])

        direct = lambda p: audit.original_cost_vector(jnp.asarray(raw)+jnp.asarray(matrix)@p)
        np.testing.assert_allclose(direct(jnp.zeros(2)), independent(np.zeros(2)), rtol=1e-13)
        fd = np.column_stack([(independent(np.eye(2)[i]*1e-6)-independent(-np.eye(2)[i]*1e-6))/2e-6
                              for i in range(2)])
        np.testing.assert_allclose(jax.jacrev(direct)(jnp.zeros(2)), fd, rtol=1e-7, atol=1e-7)

    def test_native_callbacks_match_production_selector_and_corrected_bootstrap(self):
        settings = audit.RootLossAuditSettings(inner_radius=.05)
        rho = (jnp.arange(51)+.5)/51
        state = SimpleNamespace(Er=100*(.55-rho), pressure=jnp.ones((2,51)))
        class Flux:
            def evaluate_momentum_corrected_fluxes(self, state):
                return dict(Upar_neo=jnp.stack((-1000-10*state.Er, 2000+5*state.Er)))
        runtime = SimpleNamespace(geometry=SimpleNamespace(rho_grid=rho),
            species=SimpleNamespace(charge_qp=jnp.array([-1.,1.])),
            models=SimpleNamespace(flux=SimpleNamespace(neoclassical_model=Flux())))
        callback, _, _ = audit.make_audit_callbacks(audit.ORIGINAL_CONTROL_OBJECTIVES, settings)
        for name in audit.ORIGINAL_RAW_OBJECTIVES:
            value = callback(state, runtime, audit.ORIGINAL_CONTROL_OBJECTIVES.index(name))
            expected = reverse.objective_scalar_by_index(state, runtime, 0, objective_labels=(name,),
                er_transition_rho_min=.05, er_transition_rho_max=.9, er_transition_rho_target=.514)
            np.testing.assert_allclose(value, expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(audit.ORIGINAL_RAW_OBJECTIVES[2], "bootstrap_current_softmax_abs_scaled")

    def test_summary_applies_weighted_chain_rule_to_native_rows(self):
        labels = audit.ORIGINAL_CONTROL_OBJECTIVES
        settings = audit.RootLossAuditSettings(inner_radius=.05)
        rho = (np.arange(51)+.5)/51
        er = 100*(.55-rho)
        _, observe, snapshot = audit.make_audit_callbacks(labels, settings)
        observe(SimpleNamespace(Er=er), SimpleNamespace(geometry=SimpleNamespace(rho_grid=rho)), 2.)
        values = np.zeros(len(labels))
        raw = np.array([.03,.2,.5])
        matrix = np.array([[.001,-.002],[.1,-.2],[.3,.2]])
        jacobian = np.zeros((len(labels),2))
        for name, value, gradient in zip(audit.ORIGINAL_RAW_OBJECTIVES, raw, matrix, strict=True):
            values[labels.index(name)], jacobian[labels.index(name)] = value, gradient
        result = audit.summarize_audit(residual_labels=["transport:"+name for name in labels],
            parameter_labels=["n0","vmec:RBC:1:0"], values=values, jacobian=jacobian,
            profile_cfg=dict(n0=4.21), settings=settings, snapshot=snapshot)
        expected = np.asarray(jax.jacrev(audit.original_cost_vector)(jnp.asarray(raw)))@matrix
        for index, name in enumerate(audit.ORIGINAL_COST_OBJECTIVES):
            row = next(c for c in result["comparisons"] if c["name"] == name)
            np.testing.assert_allclose(list(row["physical_parameter_gradient"].values()), expected[index])
        self.assertAlmostEqual(result["baseline_metrics"]["original_objectives"]["bootstrap_penalty"], .4)
        table = saved.candidate_rows(dict(all_finite=True, parameter_order=["n0","vmec:RBC:1:0"],
            residual_labels=["transport:"+name for name in labels], jacobian=jacobian, root_loss_audit=result),
            saved.crossing(er,rho,target=settings.target,inner_radius=settings.inner_radius,outer_radius=settings.outer_radius))
        self.assertTrue(any(row["candidate"] == "audit_root_bootstrap_cost" for row in table))
        self.assertTrue(any(row.get("strength_cost") is not None for row in table))

    def test_original_suite_cannot_reuse_er_only_forward_cache(self):
        seed = legacy_report()
        with patch.object(pilot.descent, "source_digest", return_value="same physics"):
            task = pilot.validate_seed(seed, pilot.PilotOptions(candidate="original", allow_root_creation=True))
        er_only = dict(task)
        er_only.pop("objective_suite")
        er_only.pop("original_objective_settings")
        self.assertNotEqual(pilot.forward_key(task), pilot.forward_key(er_only))
        old = analytic.report()
        with patch.object(pilot.descent, "source_digest", return_value="same physics"), self.assertRaisesRegex(ValueError,"lacks"):
            pilot.validate_seed(old, pilot.PilotOptions(candidate="bootstrap", allow_root_creation=True))

    def test_four_original_and_bootstrap_runs_start_independently_and_keep_other_dofs_fixed(self):
        evaluators = []
        def evaluator(*args):
            value = LegacyEvaluator()
            evaluators.append(value)
            return value
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed = root / "seed.json"
            seed.write_text(json.dumps(legacy_report()))
            with patch.object(pilot,"SubprocessEvaluator",side_effect=evaluator), \
                 patch.object(pilot.descent,"source_digest",return_value="physics"), \
                 patch.object(comparison,"plot_comparison"), redirect_stdout(io.StringIO()):
                comparison.main(["--ad-report",str(seed),"--out-dir",str(root/"out"),
                    "--candidates","original,bootstrap","--groups","profiles,geometry",
                    "--max-iterations","1","--initial-step",".1","--max-step",".2"])
            self.assertEqual(len(evaluators),4)
            for index, value in enumerate(evaluators):
                self.assertEqual(value.points[0],analytic.BASE)
                last = value.points[-1]
                if index%2 == 0:
                    self.assertEqual(last[6:],analytic.BASE[6:])
                else:
                    self.assertEqual(last[:6],analytic.BASE[:6])
            table = json.loads((root/"out/comparison.json").read_text())["runs"]
            self.assertEqual(len(table),4)
            self.assertTrue(all(row["final_bootstrap_penalty"] is not None for row in table))

    def test_bootstrap_does_not_stop_just_because_root_is_at_target(self):
        seed = legacy_report([1.2,*analytic.BASE[1:]])
        result = pilot.optimize(seed,LegacyEvaluator(),pilot.PilotOptions(candidate="bootstrap",allow_root_creation=True,
            max_iterations=1,initial_step=.1,max_step=.2))
        self.assertEqual(len(result["history"])-1,1)
        self.assertFalse(result["converged"])
        self.assertIn("bootstrap_cost",result["history"][-1]["component_costs"])

    def test_bootstrap_target_can_be_met_without_root_position_target(self):
        seed = legacy_report([1.4,*analytic.BASE[1:]])
        evaluator = LegacyEvaluator()
        # Floating-point representation near 0.6 must not decide this test.
        ordinary = evaluator.forward
        def forward(point):
            value = ordinary(point)
            value["metrics"]["original_objectives"]["bootstrap_current_softmax_abs_scaled"] = .09
            return value
        evaluator.forward = forward
        result = pilot.optimize(seed,evaluator,pilot.PilotOptions(candidate="bootstrap",allow_root_creation=True))
        self.assertEqual(result["status"],"bootstrap_target_met")
        self.assertEqual(len(result["history"]),1)

    def test_accepted_point_ad_keeps_native_original_observable_rows(self):
        from NEOPAX import _orchestrator, optimization
        seed = legacy_report()
        point = [1.9,*analytic.BASE[1:]]
        evaluate = Mock(return_value={})
        benchmark = SimpleNamespace(_benchmark_device_context=lambda _:nullcontext(),
            _run_realtime_geometry_optimization_api_smoke=evaluate)
        with patch.dict("sys.modules",{"benchmark_transport_reverse_ad_only":benchmark}), \
             patch.object(pilot,"ad_arguments",return_value=SimpleNamespace()), \
             patch.object(_orchestrator,"build_runtime_context",return_value=(object(),object())), \
             patch.object(optimization,"build_geometry_autodiff_context",return_value=object()):
            pilot.run_ad_task(dict(seed_report=seed,forward_task=dict(parameter_values=point)))
        self.assertEqual(evaluate.call_args.kwargs["root_audit_objective_labels"],audit.ORIGINAL_CONTROL_OBJECTIVES)
        self.assertEqual(evaluate.call_args.kwargs["parameter_values_override"],point)


if __name__ == "__main__":
    unittest.main()
