"""Saved-array analysis only; never import or execute a plasma solver."""

import ast
from contextlib import redirect_stdout
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from examples.benchmarks import analyze_er_root_saved_profiles as analysis


SETTINGS = dict(target=.514, inner_radius=.3, outer_radius=.9)
RHO = np.array([.2, .4, .6, .8])
LABELS = ["n0", "T0", "vmec:RBC:1:0"]


def point(q=.55, slope=100.):
    er = slope * (q - RHO)
    return dict(rho=RHO.tolist(), Er=er.tolist(), crossing=analysis.crossing(er, RHO, **SETTINGS))


def report(values=(2., 1., 0.), q=.55, slope=100.):
    snapshot = point(q, slope)
    target_er = snapshot["crossing"]["Er_at_target"]
    zero = .5 * (target_er / 2)**2
    gq = np.array([.01, 0., .02])
    ga = np.array([0., 20., 0.])
    ge = slope * gq + (q - SETTINGS["target"]) * ga
    gc = target_er / 4 * ge
    costs = {name: dict(cost=zero, zero_cost=zero, sign_cost=0.) for name in (analysis.LOCAL, analysis.ANNULAR)}
    return dict(all_finite=True, parameter_order=LABELS,
                residual_labels=["transport:" + name for name in (
                    "audit_root_interpolated_Er", analysis.LOCAL, analysis.ANNULAR,
                    "audit_root_nearest_downcrossing_rho")],
                jacobian=np.stack([ge, gc, gc, gq]).tolist(),
                root_loss_audit=dict(settings={**SETTINGS, "profile_trust_multiplier": 1., "geometry_ess_alpha": 1.2},
                    target=SETTINGS["target"], coordinate_scales=[2., 1., 3.],
                    terminal_Er=snapshot["Er"], terminal_rho=snapshot["rho"], terminal_time=20.,
                    experiment=dict(parameter_values=list(values), resolved_config={}),
                    baseline_metrics=dict(candidates=costs)))


class SavedRootProfileAnalysisTests(unittest.TestCase):
    def test_common_rescaling_reduces_target_field_without_moving_off_grid_root(self):
        first, second = point(), point(slope=80.)
        change = analysis.compare_profiles(first, second, SETTINGS["target"])
        self.assertAlmostEqual(first["crossing"]["root"], .55)
        self.assertAlmostEqual(first["crossing"]["fraction"], .75)
        self.assertAlmostEqual(second["crossing"]["fraction"], .75)
        self.assertAlmostEqual(change["root_shift"], 0.)
        self.assertAlmostEqual(change["crossing_motion_component"], 0.)
        self.assertAlmostEqual(change["local_slope_fraction"], 1.)
        self.assertAlmostEqual(change["local_rescaling_factor"], .8)
        self.assertLess(change["non_scaling_fraction_of_local_change"], 1e-12)

    def test_translation_at_fixed_slope_attributes_the_field_change_to_root_motion(self):
        change = analysis.compare_profiles(point(), point(q=.53), SETTINGS["target"])
        self.assertAlmostEqual(change["root_shift"], -.02)
        self.assertAlmostEqual(change["crossing_motion_fraction"], 1.)
        self.assertAlmostEqual(change["local_slope_component"], 0.)

    def test_mixed_change_closes_exactly_and_keeps_signed_cancellation(self):
        change = analysis.compare_profiles(point(), point(q=.53, slope=120.), SETTINGS["target"])
        self.assertAlmostEqual(change["crossing_motion_component"], -2.2)
        self.assertAlmostEqual(change["local_slope_component"], .52)
        self.assertAlmostEqual(change["target_Er_change"], -1.68)
        self.assertAlmostEqual(change["closure_error"], 0.)
        self.assertGreater(change["crossing_motion_fraction"], 1.)
        self.assertLess(change["local_slope_fraction"], 0.)

    def test_bracket_change_missing_root_and_moved_grid_do_not_receive_false_decompositions(self):
        self.assertFalse(analysis.compare_profiles(point(), point(q=.35), SETTINGS["target"])["decomposition_available"])
        missing = analysis.crossing(-np.ones(4), RHO, **SETTINGS)
        self.assertIsNone(missing["root"])
        self.assertIsNone(missing["bracket"])
        moved = point()
        moved["rho"] = (RHO + .001).tolist()
        self.assertFalse(analysis.compare_profiles(point(), moved, SETTINGS["target"])["decomposition_available"])

    def test_small_axis_negative_field_does_not_replace_the_annular_crossing(self):
        rho = np.array([.01, .1, .2, .4, .6, .8, .95])
        er = np.array([0., -.1, 1., 10., -10., -20., -30.])
        root = analysis.crossing(er, rho, **SETTINGS)
        self.assertEqual(root["downcrossings"], [.5])
        self.assertEqual(root["bracket"], 3)

    def test_saved_rows_compare_candidates_and_separate_amplitude_from_root_control(self):
        seed, snapshot = report(), point()
        rows = analysis.candidate_rows(seed, snapshot["crossing"])
        self.assertEqual(len(rows), 4)
        profiles = next(row for row in rows if row["candidate"] == analysis.ANNULAR and row["group"] == "profiles")
        geometry = next(row for row in rows if row["candidate"] == analysis.ANNULAR and row["group"] == "geometry")
        self.assertGreater(profiles["root_direction_alignment"], 0.)
        self.assertLess(profiles["root_direction_alignment"], 1.)
        self.assertAlmostEqual(geometry["root_direction_alignment"], 1.)
        self.assertAlmostEqual(profiles["scaled_sign_gradient_l2"], 0.)
        self.assertLess(profiles["Er_rate_crossing_motion_fraction"], 1.)
        self.assertEqual(rows[0]["root_change_per_unit_descent"], rows[2]["root_change_per_unit_descent"])

    def test_missing_crossing_is_not_interpreted_as_successful_zero_root_gradient(self):
        snapshot = analysis.crossing(np.ones(4), RHO, **SETTINGS)
        for row in analysis.candidate_rows(report(), snapshot):
            self.assertIsNone(row["root_change_per_unit_descent"])
            self.assertIsNone(row["root_direction_alignment"])

    def test_cli_reads_completed_files_and_pending_forward_without_running_a_solver(self):
        seed = report()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            run, cache, out = folder / "run", folder / "cache", folder / "analysis"
            run.mkdir()
            cache.mkdir()
            seed_path = folder / "seed.json"
            seed_path.write_text(json.dumps(seed))
            accepted_point = [1.99, 1., 0.]
            pending_point = [1.98, 1., 0.]
            accepted = report(accepted_point, q=.549, slope=99.)
            (run / "evaluation_0001_ad.json").write_text(json.dumps(accepted))
            (run / "evaluation_0001_ad_task.json").write_text('{"ignored": true}')
            snapshot = point(q=.548, slope=98.)
            initial = np.array([1., -2., 1., 1.])
            forward = dict(parameter_values=pending_point, terminal_Er=snapshot["Er"], terminal_rho=snapshot["rho"],
                           initial_state=dict(Er=initial.tolist()))
            (run / "evaluation_0002_forward.json").write_text(json.dumps(forward))
            task = dict(parameter_values=[2., 1., 0.], resolved_config={},
                        profile_contract="ad_initial_profiles_fixed_config_selected_root")
            digest = hashlib.sha256(json.dumps(task, sort_keys=True, allow_nan=False).encode()).hexdigest()
            (cache / "baseline_task.json").write_text(json.dumps(task))
            (cache / "baseline.json").write_text(json.dumps(dict(parameter_values=task["parameter_values"],
                terminal_Er=seed["root_loss_audit"]["terminal_Er"], terminal_rho=RHO.tolist(),
                initial_state=dict(Er=[1.] * 4), task_digest=digest)))
            events = [dict(event="accepted", parameters=accepted_point, iteration=1),
                      dict(event="gradient_started", parameters=pending_point, iteration=2)]
            (run / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + '\n{"incomplete":')
            with redirect_stdout(io.StringIO()):
                analysis.main(["--ad-report", str(seed_path), "--run-dir", str(run),
                               "--descent-dir", str(cache), "--out-dir", str(out)])
            result = json.loads((out / "summary.json").read_text())
            self.assertEqual([p["status"] for p in result["points"]],
                             ["baseline", "accepted", "forward_trial_AD_pending"])
            self.assertEqual(result["points"][-1]["initial_Er_annular_sign_change_indices"], [1])
            self.assertEqual(result["points"][-1]["initial_Er_max_abs_change"], 3.)
            with (out / "candidate_comparison.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 8)
            self.assertTrue((out / "crossing_cells.csv").exists())
            self.assertTrue((out / "crossing_summary.csv").exists())

    def test_reader_has_no_solver_or_jax_imports(self):
        module = ast.parse(Path(analysis.__file__).read_text())
        imports = [node.module for node in ast.walk(module) if isinstance(node, ast.ImportFrom)]
        imports += [alias.name for node in ast.walk(module) if isinstance(node, ast.Import) for alias in node.names]
        self.assertFalse(any(name and name.split(".")[0] in {"jax", "NEOPAX", "vmec_jax", "ntx"} for name in imports))


if __name__ == "__main__":
    unittest.main()
