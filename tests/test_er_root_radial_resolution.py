"""Small stored-array and mocked-worker checks; never solve plasma physics."""

import copy
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from examples.benchmarks import check_er_root_radial_resolution as radial


def inputs(directory):
    seed = Path(directory) / "input.optimized"
    seed.write_text("&INDATA\n MPOL=6, NTOR=6, RBC(0,0)=12.\n/\n")
    labels = [*radial.PROFILES, "vmec:RBC:1:0", "vmec:ZBS:1:0"]
    baseline = [4.21, 17.8, 10., 2., 1., 1., 0., 0.]
    point = [4.206, 17.7375, 10.0013, 1.9986, .99996, .99998, 0., 0.]
    config = dict(geometry=dict(n_radial=5, vmec_input_file=str(seed), mboz=18, nboz=18),
        profiles=dict(zip(radial.PROFILES, baseline[:6], strict=True)),
        neoclassical=dict(ntx_scan_rho=[.125, .5, .875], ntx_scan_n_theta=25, ntx_scan_n_zeta=25, ntx_scan_n_xi=33),
        transport_solver=dict(t_final=2., max_steps=1000, DEr=2., rtol=1e-6, radau_jacobian_reuse_mode="legacy"))
    report = dict(all_finite=True, accepted_step_limit=None, parameter_order=labels,
        optimization_start_case=dict(seed_sha256=hashlib.sha256(seed.read_bytes()).hexdigest()),
        root_loss_audit=dict(settings=dict(target=.514, inner_radius=.05, outer_radius=.9),
            terminal_rho=((np.arange(5)+.5)/5).tolist(),
            terminal_Er=(100*(.55+.001*(baseline[0]-4.)+.01/5-(np.arange(5)+.5)/5)).tolist(),
            terminal_time=2.,
            experiment=dict(resolved_config=config, parameter_values=baseline, initial_er_root_ad="jax_selected_root")))
    checkpoint = dict(state=dict(group="profiles", report=copy.deepcopy(report), history=[
        dict(iteration=0, parameters=baseline), dict(iteration=1, parameters=point)]))
    return report, checkpoint


def make_plan(report, checkpoint, grids=(5, 9)):
    with patch.object(radial.forward, "source_digest", return_value="unchanged-physics"):
        return radial.make_plan(report, checkpoint, grids, 1)


def endpoint(task):
    n = task["resolved_config"]["geometry"]["n_radial"]
    rho = (np.arange(n)+.5)/n
    root = .55 + .001*(task["parameter_values"][0]-4.) + .01/n
    return dict(task_digest=radial.forward.task_digest(task), parameter_values=task["parameter_values"],
        terminal_rho=rho.tolist(), terminal_Er=(100*(root-rho)).tolist(),
        initial_state=dict(Er=(100*(root+.01-rho)).tolist()),
        final_time=2., accepted_steps=7, attempts=9,
        metrics=dict(candidates={"audit_root_local_cost":dict(cost=2.), "audit_root_annular_cost":dict(cost=3.)}))


def save_cached(directory, task, value=None, *, wrapper=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    packet = dict(kind="forward", forward_task=task) if wrapper else task
    (directory / "evaluation_0001_forward_task.json").write_text(json.dumps(packet))
    (directory / "evaluation_0001_forward.json").write_text(json.dumps(endpoint(task) if value is None else value))


class RadialResolutionTests(unittest.TestCase):
    def test_attempt_budget_override_keeps_original_cache_keys_and_all_other_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            report, checkpoint = inputs(directory)
            ordinary = make_plan(report, checkpoint)
            with patch.object(radial.forward, "source_digest", return_value="unchanged-physics"):
                extended = radial.make_plan(report, checkpoint, [5, 9], 1, 20000)
            for old, new in zip(ordinary["tasks"], extended["tasks"], strict=True):
                if old["resolved_config"]["geometry"]["n_radial"] == 5:
                    self.assertEqual(old, new)
                    self.assertEqual(radial.forward_key(old), radial.forward_key(new))
                else:
                    self.assertEqual(new["resolved_config"]["transport_solver"]["max_steps"], 20000)
                    restored = copy.deepcopy(new)
                    restored["resolved_config"]["transport_solver"]["max_steps"] = 1000
                    self.assertEqual(restored, old)
            with self.assertRaises(ValueError):
                with patch.object(radial.forward, "source_digest", return_value="code"):
                    radial.make_plan(report, checkpoint, [5, 9], 1, 999)

    def test_only_radial_grid_changes_and_profile_vectors_remain_fixed(self):
        with tempfile.TemporaryDirectory() as directory:
            report, checkpoint = inputs(directory)
            before = copy.deepcopy((report, checkpoint))
            plan = make_plan(report, checkpoint)
            self.assertEqual(len(plan["tasks"]), 4)
            for index, task in enumerate(plan["tasks"]):
                config = copy.deepcopy(task["resolved_config"])
                config["geometry"]["n_radial"] = 5
                self.assertEqual(config, report["root_loss_audit"]["experiment"]["resolved_config"])
                self.assertEqual(task["parameter_values"], checkpoint["state"]["history"][index % 2]["parameters"])
                self.assertEqual(task["initial_er_root_ad"], "jax_selected_root")
            self.assertEqual(plan["fixed_probe_radii"], [.5, .514, .7])
            self.assertEqual((report, checkpoint), before)

    def test_wrong_case_geometry_moves_stale_seed_and_prefix_limits_fail_early(self):
        with tempfile.TemporaryDirectory() as directory:
            report, checkpoint = inputs(directory)
            for change in ("group", "vector", "config", "prefix"):
                a, b = copy.deepcopy((report, checkpoint))
                if change == "group":
                    b["state"]["group"] = "geometry"
                elif change == "vector":
                    b["state"]["history"][1]["parameters"][-1] = .01
                elif change == "config":
                    b["state"]["report"]["root_loss_audit"]["experiment"]["resolved_config"]["transport_solver"]["DEr"] = 3.
                else:
                    a["accepted_step_limit"] = 16
                with self.subTest(change=change), self.assertRaises(ValueError):
                    make_plan(a, b)
            with self.assertRaisesRegex(ValueError, "Accepted iteration"):
                with patch.object(radial.forward, "source_digest", return_value="code"):
                    radial.make_plan(report, checkpoint, [5, 9], 2)
            Path(report["root_loss_audit"]["experiment"]["resolved_config"]["geometry"]["vmec_input_file"]).write_text("new boundary")
            with self.assertRaisesRegex(ValueError, "geometry input changed"):
                make_plan(report, checkpoint)

    def test_existing_forward_wrapper_is_reused_and_completed_run_resumes_without_solves(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report, checkpoint = inputs(root)
            plan = make_plan(report, checkpoint)
            # Saved optimizer task names/candidate metadata do not change the physics.
            old = copy.deepcopy(plan["tasks"][0])
            old.update(name="optimization_trial_1", candidate="audit_root_local_cost", step_size=0.)
            save_cached(root / "old_run", old)
            runner = Mock(side_effect=lambda task, *_: endpoint(task))
            with redirect_stdout(io.StringIO()):
                rows = radial.run_plan(plan, root / "out", [root / "old_run"], runner=runner, plot=True)
            self.assertEqual(runner.call_count, 3)
            self.assertTrue(rows[0]["reused_from"].endswith("evaluation_0001_forward.json"))
            self.assertTrue((root / "out/radial_profiles.svg").is_file())
            self.assertTrue((root / "out/profile_moves.csv").is_file())
            self.assertTrue((root / "out/convergence.csv").is_file())
            result = json.loads((root / "out/n5_baseline.json").read_text())
            self.assertEqual(result["original_task_digest"], radial.forward.task_digest(old))
            runner.reset_mock()
            with redirect_stdout(io.StringIO()):
                resumed = radial.run_plan(plan, root / "out", [], runner=runner, plot=False)
            runner.assert_not_called()
            self.assertEqual(len(resumed), 4)

    def test_different_diffusion_sources_or_parameters_cannot_reuse_saved_physics(self):
        with tempfile.TemporaryDirectory() as directory:
            report, checkpoint = inputs(directory)
            plan = make_plan(report, checkpoint)
            for field in ("diffusion", "source", "profile"):
                task = copy.deepcopy(plan["tasks"][0])
                if field == "diffusion":
                    task["resolved_config"]["transport_solver"]["DEr"] = 99.
                elif field == "source":
                    task["source_digest"] = "old-source-code"
                else:
                    task["parameter_values"][0] += .01
                save_cached(Path(directory) / field, task)
                with self.subTest(field=field):
                    cache = radial.load_cache([Path(directory) / field])
                    self.assertEqual(len(cache), 1)
                    self.assertNotIn(radial.forward_key(plan["tasks"][0]), cache)

    def test_incomplete_or_wrong_grid_cached_endpoint_is_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            report, checkpoint = inputs(directory)
            task = make_plan(report, checkpoint)["tasks"][0]
            for name in ("time", "grid", "nonfinite", "initial", "digest"):
                value = endpoint(task)
                if name == "time":
                    value["final_time"] = 1.
                elif name == "grid":
                    value["terminal_rho"][0] += .01
                elif name == "nonfinite":
                    value["terminal_Er"][0] = float("nan")
                elif name == "initial":
                    value.pop("initial_state")
                else:
                    value["task_digest"] = "different task"
                folder = Path(directory) / name
                save_cached(folder, task, value)
                with self.subTest(name=name):
                    self.assertEqual(radial.load_cache([folder]), {})

    def test_plan_only_starts_no_workers_and_mismatched_output_plan_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            report, checkpoint = inputs(directory)
            plan = make_plan(report, checkpoint)
            runner = Mock(side_effect=AssertionError("No forward solve is allowed here"))
            out = Path(directory) / "out"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(radial.run_plan(plan, out, [], runner=runner, plan_only=True), [])
            runner.assert_not_called()
            self.assertFalse((out / "summary.json").exists())
            changed = copy.deepcopy(plan)
            changed["iteration"] = 99
            with self.assertRaisesRegex(ValueError, "different inputs"):
                radial.run_plan(changed, out, [], runner=runner)

    def test_changed_original_grid_baseline_stops_before_new_refinement_solves(self):
        with tempfile.TemporaryDirectory() as directory:
            report, checkpoint = inputs(directory)
            plan = make_plan(report, checkpoint)
            task = plan["tasks"][0]
            value = endpoint(task)
            value["terminal_Er"] = (np.asarray(value["terminal_Er"])+1.).tolist()
            cache_dir = Path(directory) / "old_run"
            save_cached(cache_dir, task, value)
            runner = Mock(side_effect=AssertionError("The mismatching baseline must stop refinement"))
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "original-grid baseline"):
                radial.run_plan(plan, Path(directory) / "out", [cache_dir], runner=runner, plot=False)
            runner.assert_not_called()

    def test_missing_root_and_subcell_width_are_reported_without_false_resolution_claim(self):
        rho = (np.arange(51)+.5)/51
        root = .517
        result = radial.level_width(1600*(root-rho), rho, root)
        self.assertAlmostEqual(result["interpolated_width_pm2"], 4/1600)
        self.assertEqual(result["centers_inside_pm2"], 0)
        self.assertIsNone(radial.level_width(-np.ones(51), rho, None)["interpolated_width_pm2"])

    def test_host_imports_no_physics_packages(self):
        code = ("import runpy,sys; runpy.run_path(sys.argv[1],run_name='radial_import'); "
                "assert not any(n.split('.')[0] in {'NEOPAX','jax','vmex','ntx'} for n in sys.modules)")
        result = subprocess.run([sys.executable, "-c", code, str(Path(radial.__file__).resolve())],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_worker_failure_shows_real_error_and_exit_code_in_parent_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_path = root / "worker.log"

            def failing_process(*args, **kwargs):
                kwargs["stdout"].write("Traceback (most recent call last):\nValueError: upstream shape mismatch\n")
                return SimpleNamespace(returncode=2)

            stderr = io.StringIO()
            with patch.object(radial.subprocess, "run", side_effect=failing_process), \
                 redirect_stdout(io.StringIO()), redirect_stderr(stderr), \
                 self.assertRaisesRegex(ValueError, "exit code 2"):
                radial.run_worker(dict(name="n101_baseline"), root / "task.json", root / "result.json", log_path)
            self.assertIn("ValueError: upstream shape mismatch", stderr.getvalue())
            self.assertIn("worker exit code=2", stderr.getvalue())
            self.assertIn("Traceback", log_path.read_text())

    def test_worker_value_error_preserves_traceback_instead_of_becoming_usage_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.json"
            path.write_text("{}")
            with patch.object(radial.forward, "run_forward_task", side_effect=ValueError("upstream shape mismatch")), \
                 self.assertRaisesRegex(ValueError, "upstream shape mismatch"):
                radial.main(["--worker-task", str(path), "--worker-output", str(path.with_name("result.json"))])

    def test_incomplete_rollout_diagnostic_distinguishes_budget_exhaustion_from_solver_failure(self):
        def _schedule_rollout(failed=False):
            solver = SimpleNamespace(t1=2.)
            max_total_steps = 1000
            rollout = SimpleNamespace(final_carry=SimpleNamespace(t=.031984666),
                final_step_state=SimpleNamespace(dt=1e-7), attempt_count=1000 if not failed else 27,
                accepted_count=500 if not failed else 12, failed=failed, completed=False,
                fail_code=0 if not failed else 3)
            raise RuntimeError("FD adaptive baseline did not complete")

        for failed in (False, True):
            try:
                _schedule_rollout(failed)
            except RuntimeError as exc:
                result = radial.incomplete_rollout_diagnostic(exc)
            with self.subTest(failed=failed):
                self.assertEqual(result["attempt_budget_exhausted"], not failed)
                self.assertEqual(result["failed"], failed)
                self.assertEqual(result["max_attempts"], 1000)
                self.assertEqual(result["next_dt"], 1e-7)

    def test_worker_retains_incomplete_counts_without_retrying_or_hiding_original_exception(self):
        def _schedule_rollout(task):
            solver = SimpleNamespace(t1=2.)
            max_total_steps = 1000
            rollout = SimpleNamespace(final_carry=SimpleNamespace(t=.03),
                attempt_count=1000, accepted_count=600, failed=False, completed=False, fail_code=0)
            raise RuntimeError("original incomplete rollout")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "task.json").write_text("{}")
            stderr = io.StringIO()
            with patch.object(radial.forward, "run_forward_task", side_effect=_schedule_rollout) as solve, \
                 redirect_stderr(stderr), self.assertRaisesRegex(RuntimeError, "original incomplete rollout"):
                radial.main(["--worker-task", str(root / "task.json"), "--worker-output", str(root / "result.json")])
            solve.assert_called_once()
            result = json.loads((root / "result_failure.json").read_text())
            self.assertTrue(result["attempt_budget_exhausted"])
            self.assertEqual(result["accepted_steps"], 600)
            self.assertIn("[radial-resolution-incomplete]", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
