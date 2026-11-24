"""Saved-array comparison and dry-run checks; no JAX or physical solvers."""
from contextlib import redirect_stdout
import csv
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT/"examples/benchmarks/benchmark_transport_ntss1d_ad_fd.py"
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("ntss1d_runner", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class NTSS1DComparisonRunnerTests(unittest.TestCase):
    def saved_case(self, root):
        ad = root/"ad"
        (ad/"snapshots").mkdir(parents=True)
        labels = ["softmax_Er", "bootstrap_current_softmax_abs_scaled"]
        columns = ["vmec:"+p if ":" in p else p for p in runner.PARAMETERS]
        jacobian = [[float(i+1) for i in range(8)], [float(-2*(i+1)) for i in range(8)]]
        (ad/"summary.json").write_text(json.dumps(dict(parameter_order=columns,
            residual_labels=["transport:"+label for label in labels],
            residuals=[2., 3.], jacobian=jacobian)))
        arrays = dict(initial_y=np.array([1., 2., 3.]), final_y=np.array([2., 3., 4.]),
            final_time=np.array(.1), **{"support['geometry'].r_grid": np.array([.1, .5, .9]),
            "support['geometry'].G_PS": np.array([np.nan, 0.]),
            "support['database'].D11_log": np.array([-2., -1.])})
        np.savez(ad/"snapshots/ad_terminal_saved.npz", **arrays)
        for i, parameter in enumerate(runner.PARAMETERS):
            folder = root/f"fd_{parameter.replace(':', '_')}"
            (folder/"snapshots").mkdir(parents=True)
            (folder/"summary.json").write_text(json.dumps(dict(objective_labels=labels,
                objective_values=[2., 3.], gradient_fd=[jacobian[0][i], jacobian[1][i]])))
            np.savez(folder/"snapshots/fd_baseline_saved.npz", **arrays)
        return arrays

    def test_plan_prints_nine_commands_without_creating_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            generic = runner.CONFIG.with_name(runner.CONFIG.name.replace("ntss1d.toml", "black_box.toml"))
            for config in (runner.CONFIG, generic):
                output = Path(temporary)/config.stem
                result = subprocess.run([sys.executable, str(SCRIPT), "--config", str(config),
                    "--out-dir", str(output), "--plan-only"], cwd=ROOT,
                    capture_output=True, text=True, check=True)
                self.assertEqual(len(result.stdout.splitlines()), 9)
                self.assertIn("--accepted-step-limit 16", result.stdout)
                self.assertIn("--reverse-database-interpolation-transpose-mode established", result.stdout)
                self.assertEqual(result.stdout.count("--ad-baseline-reference"), 8)
                self.assertEqual(result.stdout.count("--ad-geometry-reference"), 2)
                self.assertEqual(result.stdout.count("--geometry-fd-native-precision extended"), 2)
                self.assertEqual(result.stdout.count("--device auto"), 8)
                self.assertFalse(output.exists())
                fixed = subprocess.run([sys.executable, str(SCRIPT), "--config", str(config),
                    "--out-dir", str(output), "--plan-only", "--solver-mean-reduction", "fixed"],
                    cwd=ROOT, capture_output=True, text=True, check=True)
                self.assertEqual(fixed.stdout.count("NEOPAX_SOLVER_MEAN_REDUCTION=fixed"), 9)
                self.assertEqual(len(fixed.stdout.splitlines()), 9)
                self.assertFalse(output.exists())
                captured = subprocess.run([sys.executable, str(SCRIPT), "--config", str(config),
                    "--out-dir", str(output), "--plan-only", "--capture-forward-ir"],
                    cwd=ROOT, capture_output=True, text=True, check=True)
                self.assertEqual(captured.stdout.count("NEOPAX_ORIGINAL_FORWARD_STOP=0"), 9)
                self.assertEqual(captured.stdout.count("NEOPAX_ORIGINAL_FORWARD_DIR="), 9)
                self.assertFalse(output.exists())
            a, b = (runner.tomllib.loads(config.read_text()) for config in (runner.CONFIG, generic))
            self.assertEqual(a["neoclassical"].pop("interpolation_mode"), runner.MODE)
            self.assertEqual(a, b)

    def test_workers_disable_cache_without_changing_parent_environment(self):
        with patch.dict(runner.os.environ, dict(XLA_FLAGS="old flags", PYTHONPATH="old path",
            JAX_COMPILATION_CACHE_DIR="old cache", JAX_ENABLE_COMPILATION_CACHE="1",
            NEOPAX_SOLVER_MEAN_REDUCTION="fixed", NEOPAX_ORIGINAL_FORWARD_DIR="old capture",
            NEOPAX_ORIGINAL_FORWARD_STOP="0", NEOPAX_CAPTURE_FORWARD_IR="1"), clear=True):
            before = dict(runner.os.environ)
            env = runner.worker_environment()
            self.assertEqual(dict(runner.os.environ), before)
            for key in ("XLA_FLAGS", "PYTHONPATH", "JAX_COMPILATION_CACHE_DIR", "NEOPAX_SOLVER_MEAN_REDUCTION",
                        "NEOPAX_ORIGINAL_FORWARD_DIR", "NEOPAX_ORIGINAL_FORWARD_STOP", "NEOPAX_CAPTURE_FORWARD_IR"):
                self.assertNotIn(key, env)
            self.assertEqual(env["JAX_ENABLE_COMPILATION_CACHE"], "0")
            self.assertEqual(env["JAX_ENABLE_X64"], "true")
        self.assertNotIn("jax", sys.modules)

    def test_capture_worker_keeps_command_and_environment_but_does_not_stop_early(self):
        process = Mock()
        process.stdout = iter(["original benchmark completed\n"])
        process.wait.return_value = 0
        process.__enter__ = Mock(return_value=process)
        process.__exit__ = Mock(return_value=False)
        env = {'NEOPAX_CAPTURE_FORWARD_IR':'1', 'JAX_ENABLE_X64':'true'}
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()), \
             patch.object(runner.subprocess, 'Popen', return_value=process) as launch:
            folder = Path(temporary)/'ad'
            command = ['python', runner.AD_SCRIPT, '--accepted-step-limit', '16']
            runner.worker(command, folder, env)
            passed = launch.call_args.kwargs['env']
            self.assertEqual(passed['NEOPAX_ORIGINAL_FORWARD_DIR'], str(folder/'forward'))
            self.assertEqual(passed['NEOPAX_ORIGINAL_FORWARD_STOP'], '0')
            self.assertNotIn('NEOPAX_CAPTURE_FORWARD_IR', passed)
            self.assertEqual(launch.call_args.args[0][-len(command):], command)
            self.assertEqual(env, {'NEOPAX_CAPTURE_FORWARD_IR':'1', 'JAX_ENABLE_X64':'true'})
            # Kernel-check workers run unittest, not either benchmark entry point.
            process.stdout = iter([])
            runner.worker(['python', '-m', 'unittest'], Path(temporary)/'kernel', env)
            self.assertNotIn('NEOPAX_ORIGINAL_FORWARD_DIR', launch.call_args.kwargs['env'])

    def test_forward_capture_must_match_the_actual_reported_primal_exactly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arrays = self.saved_case(root)
            for name in ('ad', 'fd_n0'):
                directory = root/name
                forward = directory/'forward'
                forward.mkdir()
                (forward/'metadata.json').write_text(json.dumps({'stop_after_capture':False}))
                for filename in ('adaptive_stablehlo.mlir', 'adaptive_compiled_hlo.txt'):
                    (forward/filename).write_text('original compiler output')
                np.savez(forward/'setup.npz', **{'initial_carry.y':arrays['initial_y']})
                native = {k:arrays[k] for k in ('final_y','final_time')}
                np.savez(forward/'native.npz', **native)
                runner.require_forward_capture(directory, ad=name=='ad')
                native['final_time'] = np.nextafter(native['final_time'], np.inf)
                np.savez(forward/'native.npz', **native)
                with self.assertRaisesRegex(ValueError, "worker's reported primal"):
                    runner.require_forward_capture(directory, ad=name=='ad')
                (forward/'metadata.json').write_text(json.dumps({'stop_after_capture':True}))
                with self.assertRaisesRegex(ValueError, 'not a forward-only capture'):
                    runner.require_forward_capture(directory, ad=name=='ad')
        self.assertNotIn('jax', sys.modules)

    def test_saved_reference_is_exact_and_rejects_missing_ambiguous_or_changed_captures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "Expected one AD"):
                runner.prepare_ad_reference(root)
            self.saved_case(root)
            runner.prepare_ad_reference(root)
            capture = root/"ad/snapshots/ad_terminal_saved.npz"
            reference = root/"ad/baseline_reference.npz"
            self.assertEqual(reference.read_bytes(), capture.read_bytes())
            runner.prepare_ad_reference(root)
            reference.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "differs from its AD capture"):
                runner.prepare_ad_reference(root)
            (capture.parent/"ad_terminal_duplicate.npz").write_bytes(capture.read_bytes())
            with self.assertRaisesRegex(ValueError, "found 2"):
                runner.prepare_ad_reference(root)

    def test_matching_saved_arrays_produce_scientific_notation_comparisons(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.saved_case(root)
            with redirect_stdout(io.StringIO()):
                rows = runner.compare(root)
            self.assertEqual(len(rows), 16)
            self.assertTrue(all(row["comparison_baseline_matches"] for row in rows))
            self.assertTrue(all(row["relative_difference_to_fd"] == 0 for row in rows))
            with (root/"ad_vs_fd.csv").open() as stream:
                saved = list(csv.DictReader(stream))
            self.assertTrue(all("e" in row["relative_difference_to_fd"] for row in saved))

    def test_state_table_and_objective_baseline_mismatches_are_flagged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arrays = self.saved_case(root)
            folder = root/"fd_n0"
            changed = dict(arrays, initial_y=arrays["initial_y"]+.01)
            np.savez(folder/"snapshots/fd_baseline_saved.npz", **changed)
            folder = root/"fd_T0"
            changed = dict(arrays)
            changed["support['database'].D11_log"] = arrays["support['database'].D11_log"]+.001
            np.savez(folder/"snapshots/fd_baseline_saved.npz", **changed)
            folder = root/"fd_density_shape_power"
            report = json.loads((folder/"summary.json").read_text())
            report["objective_values"][0] += .1
            (folder/"summary.json").write_text(json.dumps(report))
            with redirect_stdout(io.StringIO()):
                rows = runner.compare(root)
            for row in rows:
                mismatch = row["parameter"] in ("n0", "T0") or (
                    row["parameter"] == "density_shape_power" and row["objective"] == "softmax_Er")
                self.assertEqual(row["comparison_baseline_matches"], not mismatch)

    def test_signed_zero_and_dtype_differences_are_not_hidden_by_zero_absolute_difference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arrays = self.saved_case(root)
            for name in ("n0", "T0"):
                changed = dict(arrays)
                changed["support['geometry'].G_PS"] = arrays["support['geometry'].G_PS"].copy()
                if name == "n0":
                    changed["support['geometry'].G_PS"][1] = -0.
                else:
                    changed["support['geometry'].G_PS"] = changed["support['geometry'].G_PS"].astype("float32")
                np.savez(root/f"fd_{name}/snapshots/fd_baseline_saved.npz", **changed)
            with redirect_stdout(io.StringIO()):
                rows = runner.compare(root)
            for row in rows:
                self.assertEqual(row["baseline_support_arrays_match"], row["parameter"] not in ("n0", "T0"))
                if row["parameter"] == "n0":
                    self.assertEqual(row["baseline_support_max_abs_difference"], 0.)

    def test_both_modes_run_all_columns_and_stop_on_a_mismatched_baseline(self):
        for mode, mean in ((m, r) for m in ("generic", runner.MODE) for r in ("native", "fixed")):
            for drift in (False, True):
                with self.subTest(mode=mode, mean=mean, drift=drift), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    output = root/"run"
                    config = root/"case.toml"
                    config.write_text(f'[geometry]\nvmec_lane="ad"\n[neoclassical]\n'
                                      f'flux_model="ntx_scan_runtime"\ninterpolation_mode="{mode}"\n')
                    labels = [f"objective_{i}" for i in range(13)]
                    columns = ["vmec:"+p if ":" in p else p for p in runner.PARAMETERS]
                    jacobian = [[float(i+j+1) for j in range(8)] for i in range(13)]
                    arrays = dict(initial_y=np.ones(3), final_y=np.ones(3), final_time=np.array(.1),
                                  **{"support['database'].D11_log": np.array([-1.])})
                    jobs = []

                    def worker(command, directory, env):
                        jobs.append(directory.name)
                        self.assertEqual(env["JAX_ENABLE_COMPILATION_CACHE"], "0")
                        self.assertEqual(env["NEOPAX_SOLVER_MEAN_REDUCTION"], mean)
                        if directory.name == "kernel_check":
                            return
                        (directory/"snapshots").mkdir(parents=True)
                        common = dict(database_interpolation_mode=mode,
                                      accepted_step_limit=16, vmec_response_mode="raw_block")
                        if directory.name == "ad":
                            common.update(all_finite=True, parameter_order=columns,
                                residual_labels=["transport:"+label for label in labels],
                                residuals=list(range(13)), jacobian=jacobian)
                            target = root/"ad_report.json"
                            np.savez(directory/"snapshots/ad_terminal_saved.npz", **arrays)
                        else:
                            parameter = command[command.index("--parameter")+1]
                            reference = Path(command[command.index("--ad-baseline-reference")+1])
                            self.assertTrue(reference.exists())
                            self.assertEqual("--ad-geometry-reference" in command, ":" in parameter)
                            common.update(parameter_name=parameter, objective_labels=labels,
                                objective_values=list(range(13)),
                                gradient_fd=[r[runner.PARAMETERS.index(parameter)] for r in jacobian])
                            target = root/"fd_reports"/f"{parameter.replace(':','_')}_forward_fd_frozen_linearized_summary.json"
                            target.parent.mkdir(exist_ok=True)
                            current = dict(arrays)
                            if drift:
                                current["final_time"] = np.array(.2)
                            np.savez(directory/"snapshots/fd_baseline_saved.npz", **current)
                        target.write_text(json.dumps(common))
                        if mean == "fixed":
                            (directory/"solver_mean_reduction.json").write_text(json.dumps(
                                dict(solver_mean_reduction="fixed", fixed_mean_sizes=[51,408])))

                    argv = [str(SCRIPT), "--config", str(config), "--out-dir", str(output),
                            "--solver-mean-reduction", mean]
                    with patch.object(runner, "worker", worker), patch.object(sys, "argv", argv), \
                         patch.object(runner, "AD_REPORT", root/"ad_report.json"), \
                         patch.object(runner, "FD_REPORTS", root/"fd_reports"), redirect_stdout(io.StringIO()):
                        if drift:
                            with self.assertRaisesRegex(RuntimeError, "FD baseline differs from AD"):
                                runner.main()
                        else:
                            runner.main()
                    workers = [name for name in jobs if name != "kernel_check"]
                    self.assertEqual(workers, ["ad", "fd_n0"] if drift else
                                     ["ad", *[f"fd_{p.replace(':','_')}" for p in runner.PARAMETERS]])
                    self.assertEqual("kernel_check" in jobs, mode == runner.MODE)
                    with (output/"ad_vs_fd.csv").open() as stream:
                        rows = list(csv.DictReader(stream))
                    self.assertEqual(len(rows), 13 if drift else 104)
                    self.assertTrue(all(row["database_interpolation_mode"] == mode for row in rows))
                    self.assertTrue(all(row["solver_mean_reduction"] == mean for row in rows))
                    self.assertTrue(all(row["comparison_baseline_matches"] == str(not drift) for row in rows))
                    if not drift:
                        changed=list(argv)
                        changed[changed.index('--solver-mean-reduction')+1] = 'fixed' if mean=='native' else 'native'
                        with patch.object(runner,'worker',side_effect=AssertionError('mixed-mode resume')) as worker_mock, \
                             patch.object(sys,'argv',changed),redirect_stdout(io.StringIO()), \
                             patch('sys.stderr',io.StringIO()),self.assertRaises(SystemExit):
                            runner.main()
                        worker_mock.assert_not_called()
        self.assertNotIn("jax", sys.modules)

    def test_fixed_worker_without_reduction_evidence_stops_before_fd(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            config=root/'config.toml'
            config.write_text('[geometry]\nvmec_lane="ad"\n[neoclassical]\nflux_model="ntx_scan_runtime"\ninterpolation_mode="generic"\n')
            def worker(command,folder,env):
                folder.mkdir()
                (root/'ad_report.json').write_text('{}')
                (folder/'solver_mean_reduction.json').write_text(json.dumps(
                    dict(solver_mean_reduction='fixed',fixed_mean_sizes=[])))
            argv=[str(SCRIPT),'--config',str(config),'--out-dir',str(root/'run'),
                  '--solver-mean-reduction','fixed']
            with patch.object(runner,'worker',side_effect=worker) as launched, \
                 patch.object(runner,'AD_REPORT',root/'ad_report.json'),patch.object(sys,'argv',argv), \
                 redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,'did not execute fixed'):
                runner.main()
            self.assertEqual(launched.call_count,1)
            self.assertEqual(launched.call_args.args[1].name,'ad')
        self.assertNotIn('jax',sys.modules)


if __name__ == "__main__":
    unittest.main()
