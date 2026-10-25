"""Configuration parity and small-array/mocked checks, without plasma solves."""

import ast
import copy
from contextlib import nullcontext, redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import tomllib
import unittest
from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import numpy as np

from examples.benchmarks import benchmark_er_root_candidates_optimization_start as experiment
from examples.benchmarks import _er_root_loss_audit as audit


def actual_case_without_solver_import():
    """Use actual config construction/default constants through AST extraction."""
    root = Path(__file__).resolve().parents[1]
    source = root / "examples/optimization/optimize_geometry_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport.py"
    module = ast.parse(source.read_text())
    needed = {"SEED_INPUT", "TRANSPORT_CONFIG", "DATABASE_N_THETA", "DATABASE_N_PHI", "DATABASE_N_XI",
              "TRANSPORT_FINAL_TIME", "TRANSPORT_MAX_STEPS", "MAX_REVERSE_ACCEPTED_STEPS", "SOLVER_DEVICE",
              "ESS_ALPHA", "QI_MBOZ", "QI_NBOZ", "SURFACES", "GEOMETRY_FAMILIES", "MAX_MODE_SCHEDULE",
              "ER_TRANSITION_RHO_TARGET", "ER_TRANSITION_RHO_MIN", "ER_TRANSITION_RHO_MAX"}
    selected = [node for node in module.body if (
        isinstance(node, ast.Assign) and any(isinstance(n, ast.Name) and n.id in needed for n in node.targets))
        or (isinstance(node, ast.FunctionDef) and node.name in {"transport_config", "max_mode_schedule_values"})]
    scope = dict(ROOT=root, copy=copy, np=np, argparse=__import__("argparse"), Path=Path,
                 load_config=lambda p: tomllib.loads(Path(p).read_text()))
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), scope)
    defaults = SimpleNamespace(seed_input=scope["SEED_INPUT"], database_n_theta=scope["DATABASE_N_THETA"],
        database_n_phi=scope["DATABASE_N_PHI"], database_n_xi=scope["DATABASE_N_XI"],
        transport_max_steps=scope["TRANSPORT_MAX_STEPS"], max_reverse_accepted_steps=scope["MAX_REVERSE_ACCEPTED_STEPS"],
        er_transition_rho_target=scope["ER_TRANSITION_RHO_TARGET"], er_transition_rho_min=scope["ER_TRANSITION_RHO_MIN"],
        er_transition_rho_max=scope["ER_TRANSITION_RHO_MAX"])
    scope["parser"] = lambda: SimpleNamespace(parse_args=lambda _: copy.deepcopy(defaults))
    return SimpleNamespace(**scope)


class RootCandidatesOptimizationStartTests(unittest.TestCase):
    def test_case_matches_the_actual_optimization_seed_profiles_time_and_mesh(self):
        case = actual_case_without_solver_import()
        args = experiment.parser().parse_args([])
        original = case.transport_config(case.parser().parse_args([]))
        prepare = Mock(side_effect=lambda cfg, **_: copy.deepcopy(cfg))
        cfg, settings, metadata = experiment.prepare_case(args, example=case, prepare_config=prepare)
        self.assertEqual(cfg["profiles"], original["profiles"])
        self.assertEqual(cfg["equations"], original["equations"])
        self.assertEqual(cfg["boundary"], original["boundary"])
        self.assertTrue(cfg["geometry"]["vmec_input_file"].endswith("input.QI_nfp2_initial"))
        self.assertEqual(cfg["transport_solver"]["t_final"], 2.)
        self.assertEqual(metadata["database_mesh"], [25, 25, 33])
        self.assertEqual(metadata["max_mode"], 2)
        self.assertEqual((metadata["mboz"], metadata["nboz"]), (18, 18))
        self.assertEqual(settings["inner_radius"], .05)
        self.assertEqual(settings["target"], .514)
        self.assertEqual(cfg["transport_solver"]["radau_jacobian_reuse_mode"], "legacy")
        self.assertEqual(cfg["neoclassical"]["ntx_exact_n_xi"], cfg["neoclassical"]["ntx_scan_n_xi"])
        self.assertNotIn("stop_after_accepted_steps", cfg["transport_solver"])
        prepare.assert_called_once()

    def test_absence_gate_distinguishes_all_positive_field_from_absent_crossing(self):
        settings = audit.RootLossAuditSettings(inner_radius=.05)
        rho = (np.arange(51) + .5) / 51
        negative = audit.terminal_metrics(-np.ones(51) * 10, rho, settings)
        experiment.check_no_electron_region(negative, 2., 2.)
        positive = audit.terminal_metrics(np.ones(51) * 10, rho, settings)
        self.assertIsNone(positive["nearest_downcrossing"])
        self.assertEqual(positive["inner_positive_fraction"], 1.)
        with self.assertRaisesRegex(ValueError, "already has positive Er"):
            experiment.check_no_electron_region(positive, 2., 2.)
        with self.assertRaisesRegex(ValueError, "final time"):
            experiment.check_no_electron_region(negative, 1., 2.)

    def test_missing_root_has_a_mean_field_creation_gradient_without_a_fake_root_gradient(self):
        settings = audit.RootLossAuditSettings(inner_radius=.05)
        rho = (jnp.arange(51) + .5) / 51
        labels = audit.ROOT_CREATION_OBJECTIVES
        evaluate, observe, snapshot = audit.make_audit_callbacks(labels, settings)
        runtime = SimpleNamespace(geometry=SimpleNamespace(rho_grid=rho))

        def values(p):
            state = SimpleNamespace(Er=-jnp.ones(51) * 10 + p[0])
            return jnp.stack([evaluate(state, runtime, i) for i in range(len(labels))])

        p = jnp.array([1.])
        observe(SimpleNamespace(Er=-jnp.ones(51) * 9), runtime, 2.)
        jac = np.asarray(jax.jacrev(values)(p))
        self.assertEqual(jac[-1, 0], 1.)
        self.assertEqual(jac[-2, 0], 0.)
        summary = audit.summarize_audit(residual_labels=["transport:" + n for n in labels],
            parameter_labels=["n0"], values=np.asarray(values(p)), jacobian=jac,
            profile_cfg={"n0": 1.}, settings=settings, snapshot=snapshot)
        self.assertFalse(summary["nearest_downcrossing_available"])
        for item in summary["comparisons"]:
            group = item["groups"]["profiles"]
            self.assertIsNone(group["predicted_root_change_per_unit_descent"])
            self.assertGreater(group["predicted_inner_mean_Er_change_per_unit_descent"], 0.)

    def test_worker_uses_matching_geometry_context_and_verified_terminal_gate(self):
        from NEOPAX import optimization as opt, _orchestrator as orchestrator
        from examples.optimization import optimize_profiles_er_root_candidate as pilot
        case = actual_case_without_solver_import()
        cli = experiment.parser().parse_args([])
        cfg, settings, metadata = experiment.prepare_case(cli, example=case, prepare_config=lambda c, **_: c)
        runtime, state = object(), object()
        specs = (SimpleNamespace(family="RBC", m=1, n=0), SimpleNamespace(family="ZBS", m=1, n=0))
        args = SimpleNamespace()

        def evaluate(**kwargs):
            rho = (np.arange(51) + .5) / 51
            kwargs["terminal_observer_check"](dict(Er=-np.ones(51), rho=rho, final_time=2.))
            return {"mock_ad": True}

        ad = Mock(side_effect=evaluate)
        bench = SimpleNamespace(_profile_cfg_scalar_value=lambda c, n: float(c.get(n, 1.)),
            _benchmark_device_context=lambda _: nullcontext(), _run_realtime_geometry_optimization_api_smoke=ad)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), \
             patch.object(experiment, "prepare_case", return_value=(cfg, settings, metadata)), \
             patch.object(opt, "build_geometry_autodiff_context", return_value="context") as context, \
             patch.object(opt, "vmex_boundary_parameterization", return_value=SimpleNamespace(specs=specs)) as parameterization, \
             patch.object(orchestrator, "build_runtime_context", return_value=(runtime, state)) as build, \
             patch.object(pilot, "ad_arguments", return_value=args), \
             patch.dict("sys.modules", {"benchmark_transport_reverse_ad_only": bench}):
            out = Path(directory)
            experiment.run_ad(cli, out)
            result = json.loads((out / "ad_summary.json").read_text())
            self.assertTrue(result["baseline_no_electron_region_verified"])
            self.assertTrue((out / "baseline_forward_observation.json").exists())
            self.assertEqual(context.call_args.kwargs["mboz"], 18)
            self.assertEqual(parameterization.call_args.kwargs["max_mode"], 2)
            self.assertEqual(ad.call_args.kwargs["root_audit_objective_labels"], audit.ROOT_CREATION_OBJECTIVES)
            self.assertFalse(ad.call_args.kwargs["write_report"])
            self.assertIs(ad.call_args.kwargs["baseline_state"], state)
            build.assert_called_once()

    def test_reused_ad_case_rejects_silent_physics_overrides_before_any_worker(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit), patch.object(experiment.subprocess, "Popen") as start:
            experiment.main(["--ad-report", "not_read.json", "--t-final", "20"])
        start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
