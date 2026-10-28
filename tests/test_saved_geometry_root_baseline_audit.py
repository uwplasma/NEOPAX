"""Read small saved arrays/HDF5 only; no plasma solves or physics AD."""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np

from examples.benchmarks import audit_saved_geometry_root_baseline as audit


SETTINGS = dict(target=.514, inner_radius=.05, outer_radius=.9)
RHO = (np.arange(51)+.5)/51


def write_reference(directory, er):
    path = Path(directory) / "optimized/transport/transport_solution.h5"
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as stream:
        stream["rho"] = RHO
        stream["ts"] = [0., 2., np.inf]
        stream["Er"] = np.stack((er*.5, er, np.full_like(er, np.nan)))
        stream["final_time"] = 2.
    return path


class SavedGeometryBaselineAuditTests(unittest.TestCase):
    def test_recorded_input_summary_and_saved_transport_input_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed, summary = root / "input.optimized", root / "optimization_summary.json"
            seed.write_text("RBC(0,0)=12.\n")
            summary.write_text("{}")
            transport_input = root / "optimized/transport/input.QI_neopax_geometry_full_transport_optimized"
            transport_input.parent.mkdir(parents=True)
            transport_input.write_bytes(seed.read_bytes())
            digest = hashlib.sha256(seed.read_bytes()).hexdigest()
            manifest = dict(resolved_config=dict(geometry=dict(vmec_input_file=str(seed))),
                seed_sha256=digest, saved_geometry_result=dict(accepted_geometry_input=str(seed),
                accepted_geometry_sha256=digest, summary_path=str(summary),
                summary_sha256=hashlib.sha256(summary.read_bytes()).hexdigest()))
            result = audit.source_provenance(manifest, root)
            self.assertTrue(result["all_recorded_sources_unchanged"])
            self.assertTrue(result["files"]["saved_transport_input"]["matches_current_baseline_input_bytes"])

            # An unchanged path does not make a cached baseline the same case.
            seed.write_text("RBC(0,0)=13.\n")
            result = audit.source_provenance(manifest, root)
            self.assertFalse(result["all_recorded_sources_unchanged"])
            self.assertFalse(result["files"]["baseline_input"]["matches_recorded"])
            self.assertFalse(result["files"]["accepted_geometry_input"]["matches_recorded"])
            self.assertFalse(result["files"]["saved_transport_input"]["matches_current_baseline_input_bytes"])
            self.assertTrue(result["files"]["summary_path"]["matches_recorded"])

    def test_missing_or_unrecorded_sources_do_not_establish_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.input"
            manifest = dict(resolved_config=dict(geometry=dict(vmec_input_file=str(path))))
            result = audit.source_provenance(manifest, directory)
            self.assertIsNone(result["all_recorded_sources_unchanged"])
            self.assertFalse(result["files"]["baseline_input"]["available"])
            self.assertIsNone(result["files"]["saved_transport_input"]["matches_current_baseline_input_bytes"])
            manifest["seed_sha256"] = "recorded"
            self.assertFalse(audit.source_provenance(manifest, directory)["all_recorded_sources_unchanged"])

    def test_summary_changes_are_distinguished_from_input_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed, summary = root / "input.optimized", root / "optimization_summary.json"
            seed.write_text("unchanged input")
            summary.write_text("old summary")
            manifest = dict(resolved_config=dict(geometry=dict(vmec_input_file=str(seed))),
                seed_sha256=hashlib.sha256(seed.read_bytes()).hexdigest(),
                saved_geometry_result=dict(summary_path=str(summary),
                    summary_sha256=hashlib.sha256(summary.read_bytes()).hexdigest()))
            summary.write_text("new summary")
            result = audit.source_provenance(manifest, root)
            self.assertFalse(result["all_recorded_sources_unchanged"])
            self.assertTrue(result["files"]["baseline_input"]["matches_recorded"])
            self.assertFalse(result["files"]["summary_path"]["matches_recorded"])

    def test_saved_electron_region_and_new_ion_case_are_distinguished(self):
        with tempfile.TemporaryDirectory() as directory:
            write_reference(directory, 10*(.6-RHO))
            observed = dict(final_time=2., terminal_rho=RHO, terminal_Er=-np.ones(51))
            result = audit.compare_saved_transport(directory, observed, SETTINGS)
            self.assertFalse(result["matches"])
            self.assertTrue(result["same_time"])
            self.assertAlmostEqual(result["saved_field"]["nearest_downcrossing"], .6)
            self.assertIsNone(result["baseline_field"]["nearest_downcrossing"])
            self.assertGreater(result["max_abs_Er_difference"], 1.)

    def test_identical_fields_match_but_different_final_time_does_not(self):
        with tempfile.TemporaryDirectory() as directory:
            er = 10*(.6-RHO)
            write_reference(directory, er)
            observed = dict(final_time=2., terminal_rho=RHO, terminal_Er=er)
            result = audit.compare_saved_transport(directory, observed, SETTINGS)
            self.assertTrue(result["matches"])
            self.assertEqual(result["max_abs_Er_difference"], 0.)
            observed["final_time"] = 1.
            self.assertFalse(audit.compare_saved_transport(directory, observed, SETTINGS)["matches"])

    def test_runtime_defaults_differ_from_the_declared_optimization_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.optimized"
            path.write_text("&INDATA\n MPOL=6\n NTOR=6\n/")
            geometry = dict(vmec_input_file=str(path))
            manifest = dict(resolved_config=dict(geometry=geometry), mboz=18, nboz=18)
            result = audit.runtime_resolution(manifest)
            self.assertEqual(result["effective"], dict(mboz=36, nboz=11))
            self.assertFalse(result["matches_declared"])
            geometry.update(mboz=18, nboz=18)
            self.assertTrue(audit.runtime_resolution(manifest)["matches_declared"])

    def test_all_positive_field_has_no_downcrossing_but_is_not_all_negative(self):
        result = audit.field_summary(np.ones(51), RHO, SETTINGS)
        self.assertIsNone(result["nearest_downcrossing"])
        self.assertEqual(result["inner_positive_fraction"], 1.)
        self.assertEqual(result["outer_negative_fraction"], 0.)

    def test_missing_reference_is_reported_without_starting_any_solver(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(audit.compare_saved_transport(directory, {}, SETTINGS)["available"])

    def test_cli_works_with_manifest_and_observation_before_ad_finishes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_reference(root, 10*(.6-RHO))
            baseline = root / "baseline"
            baseline.mkdir()
            path = root / "input.optimized"
            path.write_text("&INDATA\n MPOL=6\n NTOR=6\n/")
            (baseline / "case_manifest.json").write_text(json.dumps(dict(
                resolved_config=dict(geometry=dict(vmec_input_file=str(path))), mboz=18, nboz=18, settings=SETTINGS)))
            (baseline / "baseline_forward_observation.json").write_text(json.dumps(dict(
                final_time=2., terminal_rho=RHO.tolist(), terminal_Er=(-np.ones(51)).tolist())))
            with redirect_stdout(io.StringIO()):
                audit.main(["--geometry-result-dir", str(root), "--baseline-dir", str(baseline)])
            result = json.loads((baseline / "saved_geometry_baseline_audit.json").read_text())
            self.assertIsNone(result["source_provenance"]["all_recorded_sources_unchanged"])
            self.assertFalse(result["runtime_resolution"]["matches_declared"])
            self.assertFalse(result["terminal_comparison"]["matches"])
            self.assertFalse((baseline / "ad_summary.json").exists())

    def test_audit_imports_no_physics_packages(self):
        code = ("import runpy,sys; runpy.run_path(sys.argv[1],run_name='audit_import'); "
                "assert not any(n.split('.')[0] in {'NEOPAX','jax','vmex','ntx'} for n in sys.modules)")
        result = subprocess.run([sys.executable, "-c", code, str(Path(audit.__file__).resolve())],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
