"""Deck, file-selection and comparison checks; never run GKX or VMEC."""

from contextlib import redirect_stdout
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch

import numpy as np

from examples.optimization import run_gkx_initial_optimized_adiabatic as example


def fixture(folder):
    for label in ("initial", "optimized"):
        directory = folder / label
        directory.mkdir()
        (directory / f"wout_QI_neopax_geometry_full_transport_{label}.nc").write_bytes(label.encode())
    template = folder / "template.toml"
    template.write_text('''schema_version = 1
[[species]]
name = "ion"
charge = 1.0
mass = 1.0
tprim = 99.0
fprim = 99.0
[geometry]
model = "miller"
geometry_file = "wrong.eik.nc"
alpha = 1.2
[grid]
Nx = 96
Ny = 96
Nz = 48
[physics]
linear = true
nonlinear = false
electrostatic = false
electromagnetic = true
adiabatic_electrons = false
adiabatic_ions = true
beta = 0.1
[terms]
apar = 1.0
bpar = 1.0
[time]
t_max = 10.0
[run]
steps = 200
[init]
init_file = "unrelated.restart.nc"
''', encoding="utf-8")
    return [str(folder), "--template", str(template)]


class AdiabaticComparisonTests(unittest.TestCase):
    def test_prepare_uses_two_saved_equilibria_and_direct_s_with_matching_physics(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            argv = fixture(folder)
            with patch.object(example.scan, "_run_deck", side_effect=AssertionError("GKX forbidden")), \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(example.main([*argv, "--prepare-only"]), 0)
            out = folder / "gkx_adiabatic_comparison/s_0p5_LT_3_Ln_0_alpha_0"
            manifest = json.loads((out / "case_manifest.json").read_text())
            decks = []
            for point in manifest["points"]:
                deck = tomllib.loads(Path(point["deck"]).read_text())
                self.assertEqual(deck["geometry"]["torflux"], .5)  # Never .25.
                self.assertEqual(deck["geometry"]["alpha"], 0.)
                self.assertEqual(deck["geometry"]["model"], "vmec")
                self.assertNotIn("geometry_file", deck["geometry"])
                self.assertEqual(deck["geometry"]["vmec_file"], point["wout"])
                self.assertIn(point["equilibrium"], Path(point["wout"]).stem)
                self.assertEqual(deck["species"][0]["tprim"], 3.)
                self.assertEqual(deck["species"][0]["fprim"], 0.)
                self.assertTrue(deck["species"][0]["kinetic"])
                self.assertTrue(deck["physics"]["adiabatic_electrons"])
                self.assertFalse(deck["physics"]["adiabatic_ions"])
                self.assertTrue(deck["physics"]["nonlinear"])
                self.assertFalse(deck["physics"]["linear"])
                self.assertTrue(deck["physics"]["electrostatic"])
                self.assertFalse(deck["physics"]["electromagnetic"])
                self.assertEqual(deck["physics"]["beta"], 0.)
                self.assertEqual(deck["terms"]["apar"], 0.)
                self.assertEqual(deck["terms"]["bpar"], 0.)
                self.assertNotIn("steps", deck["run"])
                self.assertNotIn("init_file", deck["init"])
                self.assertEqual(deck["time"]["run_to"], "t_max")
                self.assertEqual(deck["time"]["t_max"], 700.)
                normalized = copy.deepcopy(deck)
                normalized["geometry"].pop("vmec_file")
                normalized["output"].pop("path")
                decks.append(normalized)
            self.assertEqual(decks[0], decks[1])

    def test_overrides_and_worker_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            argv = fixture(folder)
            with patch.object(example.scan, "_run_deck") as run, \
                 patch.object(example, "_write_comparison") as compare, redirect_stdout(io.StringIO()):
                example.main([*argv, "--s", ".3", "--LT", "4", "--Ln", ".2", "--alpha", ".7", "--t-max", "123"])
            self.assertEqual(run.call_count, 2)
            self.assertEqual([call.args[0].parent.name for call in run.call_args_list], ["initial", "optimized"])
            self.assertTrue(all(call.kwargs["t_max"] == 123. for call in run.call_args_list))
            for point in compare.call_args.args[0]:
                deck = tomllib.loads(Path(point["deck"]).read_text())
                self.assertEqual(deck["geometry"]["torflux"], .3)
                self.assertEqual(deck["geometry"]["alpha"], .7)
                self.assertEqual(deck["species"][0]["tprim"], 4.)
                self.assertEqual(deck["species"][0]["fprim"], .2)

    def test_missing_initial_never_falls_back_to_optimized(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "optimized").mkdir()
            (folder / "optimized/wout_optimized.nc").touch()
            with self.assertRaisesRegex(ValueError, "initial-wout"):
                example._find_wout(folder, "initial", None)

    def test_reuse_requires_same_wout_content_and_input_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            argv = fixture(folder)
            with redirect_stdout(io.StringIO()):
                example.main([*argv, "--prepare-only"])
                example.main([*argv, "--prepare-only"])
            wout = folder / "optimized/wout_QI_neopax_geometry_full_transport_optimized.nc"
            wout.write_bytes(b"different optimized geometry")
            with patch.object(example.scan, "_run_deck", side_effect=AssertionError("GKX forbidden")), \
                 self.assertRaisesRegex(ValueError, "inputs differ"):
                example.main([*argv, "--prepare-only"])

    def test_plot_only_never_dispatches_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            argv = fixture(folder)
            with redirect_stdout(io.StringIO()):
                example.main([*argv, "--prepare-only"])
            with patch.object(example.scan, "_run_deck", side_effect=AssertionError("GKX forbidden")), \
                 patch.object(example.scan, "_prepare_deck", side_effect=AssertionError("deck mutation forbidden")), \
                 patch.object(example, "_write_comparison") as compare:
                example.main([*argv, "--plot-only"])
            compare.assert_called_once()

    def test_statistics_weight_adaptive_times_rather_than_sample_counts(self):
        stats = example._late_statistics([0., .1, .2, 1.], [0., .1, .2, 1.], .5)
        self.assertAlmostEqual(stats["average_window_start"], .5)
        self.assertAlmostEqual(stats["ion_heat_flux_mean"], .75)
        self.assertAlmostEqual(stats["ion_heat_flux_std"], .5/np.sqrt(12.))

    def test_writes_labeled_synthetic_comparison_without_solver(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            argv = fixture(folder)
            with redirect_stdout(io.StringIO()):
                example.main([*argv, "--prepare-only"])
            out = folder / "gkx_adiabatic_comparison/s_0p5_LT_3_Ln_0_alpha_0"
            points = json.loads((out / "case_manifest.json").read_text())["points"]
            with patch.object(example.scan, "_read_heat_flux", side_effect=[
                (np.arange(6.), np.full(6, 4.)), (np.arange(6.), np.full(6, 2.))
            ]), redirect_stdout(io.StringIO()):
                example._write_comparison(points, out, .2)
            report = json.loads((out / "comparison.json").read_text())
            self.assertEqual(report["optimized_over_initial"], .5)
            self.assertEqual(report["optimized_minus_initial"], -2.)
            with (out / "heat_flux_summary.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["equilibrium"] for row in rows], ["initial", "optimized"])
            self.assertTrue((out / "heat_flux_comparison.png").is_file())


if __name__ == "__main__":
    unittest.main()
