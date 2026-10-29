"""Mocked wout diagnostics and small arrays; no VMEC, NTX or transport solves."""

from contextlib import redirect_stdout
import copy
from dataclasses import dataclass
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from examples.optimization import plot_initial_er_root_from_wout as example


@dataclass(frozen=True)
class State:
    density: np.ndarray
    pressure: np.ndarray
    Er: np.ndarray


class WoutErBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.rho = np.asarray([.2, .5, .8])
        self.er = np.asarray([6., 2., -3.])
        self.mask = np.asarray([True, True, True])
        self.state = State(np.ones((2, 3)), np.full((2, 3), 3.), np.full(3, 99.))
        self.corrected = Mock(return_value={"Upar_neo": np.asarray([[1., 2., 3.], [3., 1., 7.]])})
        self.neo = SimpleNamespace(evaluate_momentum_corrected_fluxes=self.corrected)
        self.runtime = SimpleNamespace(
            geometry=SimpleNamespace(rho_grid=self.rho),
            species=SimpleNamespace(charge_qp=np.asarray([-1., 1.])),
            models=SimpleNamespace(flux=SimpleNamespace(neoclassical_model=self.neo)),
        )
        self.config = {"general": {"mode": "ambipolarity"}}

    def solve(self, *, legacy=False, **kwargs):
        function = example.solve_initial_er_root_from_wout if legacy else example.solve_initial_er_and_bootstrap_from_wout
        with patch.object(example, "_config_for_wout", return_value=self.config), \
             patch.object(example, "build_runtime_context", return_value=(self.runtime, self.state)) as build, \
             patch.object(example, "initial_er_selected_root_profile", return_value=(self.er, self.mask)) as root:
            result = function(Path("wout.nc"), config_path=Path("profiles.toml"),
                              boozer_path=Path("boozmn.nc"), **kwargs)
        build.assert_called_once_with(self.config)
        root.assert_called_once_with(self.state, config=self.config, runtime=self.runtime)
        return result

    def test_current_uses_selected_er_in_same_state_and_optimizer_units(self):
        rho, er, current, mask = self.solve()
        rooted = self.corrected.call_args.args[0]
        self.corrected.assert_called_once()
        self.assertIs(rooted.density, self.state.density)
        self.assertIs(rooted.pressure, self.state.pressure)
        np.testing.assert_array_equal(rooted.Er, self.er)
        np.testing.assert_array_equal(self.state.Er, [99., 99., 99.])
        # Independent physical conversion: e * 1e20 * (Upar_ion - Upar_e).
        expected_a_m2 = 1.602176634e-19 * 1e20 * np.asarray([2., -1., 4.])
        np.testing.assert_allclose(current * 1e5, expected_a_m2, rtol=1e-6)
        np.testing.assert_array_equal(rho, self.rho)
        np.testing.assert_array_equal(er, self.er)
        np.testing.assert_array_equal(mask, self.mask)

    def test_corrected_upar_fallback_and_both_species_layouts(self):
        values = np.asarray([[1., 2., 3.], [3., 1., 7.]])
        for payload in ({"Upar_neo": values, "Upar": np.zeros((2, 3))},
                        {"Upar": values.T}, SimpleNamespace(Upar_neo=values.T)):
            with self.subTest(payload=type(payload)):
                self.corrected.return_value = payload
                _, _, current, _ = self.solve()
                np.testing.assert_allclose(current * 1e5,
                                           16.02176634 * np.asarray([2., -1., 4.]), rtol=1e-6)

    def test_failed_root_is_not_reported_as_bootstrap_at_fallback_er(self):
        self.mask[1] = False
        _, _, current, mask = self.solve()
        self.assertTrue(np.isnan(current[1]))
        self.assertTrue(np.all(np.isfinite(current[[0, 2]])))
        np.testing.assert_array_equal(mask, self.mask)

    def test_all_failed_roots_and_nonfinite_current_are_errors(self):
        self.mask[:] = False
        with self.assertRaisesRegex(RuntimeError, "No finite ambipolar roots"):
            self.solve()
        self.corrected.assert_not_called()
        self.mask[:] = True
        self.corrected.return_value = {"Upar_neo": np.full((2, 3), np.nan)}
        with self.assertRaisesRegex(FloatingPointError, "Nonfinite momentum-corrected"):
            self.solve()

    def test_missing_corrected_evaluator_does_not_use_generic_flux(self):
        self.runtime.models.flux = Mock(spec=["__call__"])
        with self.assertRaisesRegex(ValueError, "evaluate_momentum_corrected_fluxes"):
            self.solve()
        self.runtime.models.flux.assert_not_called()

    def test_legacy_er_interface_never_evaluates_bootstrap(self):
        rho, er, mask = self.solve(legacy=True)
        self.corrected.assert_not_called()
        np.testing.assert_array_equal(rho, self.rho)
        np.testing.assert_array_equal(er, self.er)
        np.testing.assert_array_equal(mask, self.mask)

    def test_config_uses_requested_wout_and_preserves_profiles_without_vmec_solve(self):
        config = {"general": {"mode": "transport"},
                  "geometry": dict(backend="vmec_jax_booz_xform_jax", vmec_input_file="wrong",
                                   vmec_lane="forward", vmec_param_delta=1., n_radial=51),
                  "profiles": {"n0": 4.21, "T0": 17.8}, "species": {"names": ["e", "D"]}}
        original = copy.deepcopy(config)
        with patch.object(example, "load_config", return_value=config):
            actual = example._config_for_wout(Path("profiles.toml"), Path("chosen.nc"), Path("chosen_booz.nc"))
        self.assertEqual(config, original)
        self.assertEqual(actual["profiles"], original["profiles"])
        self.assertEqual(actual["species"], original["species"])
        self.assertEqual(actual["general"]["mode"], "ambipolarity")
        self.assertEqual(actual["geometry"], dict(n_radial=51, vmec_file="chosen.nc", boozer_file="chosen_booz.nc"))

    def test_cli_writes_both_profiles_with_physical_units_and_creates_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            wout, config, boozer = (folder / name for name in ("wout.nc", "profiles.toml", "boozmn.nc"))
            for path in (wout, config, boozer):
                path.touch()
            output = folder / "new/er.png"
            bootstrap = folder / "other/bootstrap.png"
            # Analytic saved values only; real plotting and CSV writing, no runtime.
            current = np.asarray([.01, -.02, .03])
            mask = np.asarray([True, False, True])
            current[1] = np.nan
            with patch.object(example, "solve_initial_er_and_bootstrap_from_wout",
                              return_value=(self.rho, self.er, current, mask)) as solve, \
                 patch.object(example, "_generate_boozer_file", side_effect=AssertionError("No Boozer solves")), \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(example.main([str(wout), "--config", str(config), "--boozer-file", str(boozer),
                                               "--out", str(output), "--bootstrap-out", str(bootstrap)]), 0)
            solve.assert_called_once_with(wout.resolve(), config_path=config.resolve(),
                                          boozer_path=boozer.resolve(), include_bootstrap=True)
            self.assertGreater(output.stat().st_size, 0)
            self.assertGreater(bootstrap.stat().st_size, 0)
            old = np.genfromtxt(output.with_suffix(".csv"), delimiter=",", names=True)
            self.assertEqual(old.dtype.names, ("rho", "Er", "finite_mask"))
            np.testing.assert_array_equal(old["Er"], self.er)
            data = np.genfromtxt(bootstrap.with_suffix(".csv"), delimiter=",", names=True)
            np.testing.assert_allclose(data["Jboot_A_m2"], [1000., np.nan, 3000.], equal_nan=True)
            np.testing.assert_allclose(data["Jboot_kA_m2"], [1., np.nan, 3.], equal_nan=True)
            np.testing.assert_allclose(data["Jboot_scaled_1e5_A_m2"], current, equal_nan=True)
            self.assertTrue(np.isnan(data["Er_kV_m"][1]))
            np.testing.assert_array_equal(data["finite_mask"], mask)

    def test_cli_er_only_keeps_existing_default_output_names(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            wout, config, boozer = (folder / name for name in ("wout_test.nc", "profiles.toml", "boozmn.nc"))
            for path in (wout, config, boozer):
                path.touch()
            with patch.object(example, "_config_for_wout", return_value=self.config), \
                 patch.object(example, "build_runtime_context", return_value=(self.runtime, self.state)), \
                 patch.object(example, "initial_er_selected_root_profile", return_value=(self.er, self.mask)), \
                 patch.object(example, "plot_er_profile"), \
                 redirect_stdout(io.StringIO()):
                example.main([str(wout), "--config", str(config), "--boozer-file", str(boozer), "--no-bootstrap"])
            self.corrected.assert_not_called()
            self.assertTrue((folder / "wout_test_initial_er_root.csv").is_file())
            self.assertFalse((folder / "wout_test_initial_er_root_bootstrap_current.csv").exists())


class WoutPressureProfileTests(unittest.TestCase):
    @staticmethod
    def write_wout(path, *, b0=9., pressure=None, field_variable="b0"):
        from netCDF4 import Dataset

        if pressure is None:
            pressure = np.asarray([120000., 90000., 60000., 30000., 0.])
        with Dataset(path, "w") as dataset:
            dataset.createDimension("radius", len(pressure))
            dataset.createVariable("presf", "f8", ("radius",))[:] = pressure
            # Negative toroidal flux must still give increasing s=phi/phi_edge.
            dataset.createVariable("phi", "f8", ("radius",))[:] = np.linspace(0., -2., len(pressure))
            dataset.createVariable(field_variable, "f8")[...] = b0

    def test_field_scaling_quasineutrality_and_total_pressure_on_correct_s_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wout.step780.nc"
            for field in (9., -4.5):
                with self.subTest(field=field):
                    self.write_wout(path, b0=field)
                    config = example.config_with_wout_pressure({"geometry": {"n_radial": 4}}, path)
                    rho = (np.arange(4) + .5) / 4
                    expected_p = 120000. * (1. - rho**2)
                    density = np.asarray(config["profiles"]["density"])
                    temperature = np.asarray(config["profiles"]["temperature"])
                    ne_axis = 4.21e20 * abs(field) / 9.
                    self.assertEqual(config["species"]["names"], ["e", "H"])
                    self.assertEqual(config["species"]["mass_mp"][1], 1.)
                    self.assertEqual(config["species"]["charge_qp"], [-1., 1.])
                    np.testing.assert_array_equal(density[0], density[1])
                    np.testing.assert_array_equal(temperature[0], temperature[1])
                    np.testing.assert_allclose(density[0], ne_axis * np.sqrt(1. - rho**2))
                    np.testing.assert_allclose(np.sum(density * temperature, axis=0) * 1.602176634e-19,
                                               expected_p, rtol=1e-14)
                    self.assertAlmostEqual(config["wout_pressure"]["ne_axis_m3"] / ne_axis, 1.)
                    # Exercise the actual prescribed-profile state conversion,
                    # without geometry construction, models or physics solves.
                    from NEOPAX._orchestrator import _build_state
                    state = _build_state(config, SimpleNamespace(r_grid=rho),
                                         SimpleNamespace(number_species=2, charge_qp=np.asarray([-1., 1.])))
                    np.testing.assert_allclose(np.asarray(state.density), density / 1e20)
                    np.testing.assert_allclose(np.sum(np.asarray(state.pressure), axis=0) * 1e23 * 1.602176634e-19,
                                               expected_p, rtol=1e-14)

    def test_pressure_mode_replaces_old_species_settings_without_mutating_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wout.step780.nc"
            self.write_wout(path)
            original = dict(geometry=dict(n_radial=3, rho_edge=.8),
                            profiles=dict(n0=100.), species=dict(names=["e", "D", "T", "He"]),
                            boundary=dict(density=dict(right=dict(value=dict(e=.3, D=.1, T=.2)))),
                            equations=dict(toggle_temperature=[True] * 4),
                            turbulence=dict(flux_model="turbulent_power_analytical", chi_density=[1.] * 4),
                            sources=dict(temperature=["dt_reaction"]), optimization=dict(species_profile_quasineutral=True),
                            neoclassical=dict(flux_model="ntx_exact_lij_runtime"))
            before = copy.deepcopy(original)
            config = example.config_with_wout_pressure(original, path)
            self.assertEqual(original, before)
            self.assertEqual(config["boundary"], {})
            self.assertEqual(config["sources"], {})
            self.assertEqual(config["turbulence"], {"flux_model": "none"})
            self.assertNotIn("equations", config)
            self.assertNotIn("optimization", config)
            self.assertEqual(config["neoclassical"], original["neoclassical"])
            self.assertEqual(np.asarray(config["profiles"]["density"]).shape, (2, 3))
            np.testing.assert_allclose(config["wout_pressure"]["rho"], [(.5 + i) * .8 / 3 for i in range(3)])

    def test_profiles_only_exports_reusable_toml_without_any_physics_or_boozer(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            path = folder / "wout.step780.nc"
            self.write_wout(path, b0=3.)
            template = folder / "base.toml"
            template.write_text('[geometry]\nn_radial = 4\n[profiles]\nn0 = 99.0\n', encoding="utf-8")
            out = folder / "new/step780.png"
            forbidden = AssertionError("Profiles-only must not run any physics or Boozer transforms")
            with patch.object(example, "build_runtime_context", side_effect=forbidden), \
                 patch.object(example, "_resolve_boozer_file", side_effect=forbidden), \
                 patch.object(example, "initial_er_selected_root_profile", side_effect=forbidden), \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(example.main([str(path), "--config", str(template), "--out", str(out),
                                               "--wout-pressure", "--profiles-only"]), 0)
            generated = example.load_config(folder / "new/step780_profiles.toml")
            self.assertEqual(generated["species"]["n_species"], 2)
            self.assertEqual(generated["profiles"]["model"], "prescribed")
            self.assertEqual(generated["geometry"]["vmec_file"], str(path.resolve()))
            self.assertAlmostEqual(generated["wout_pressure"]["ne_axis_m3"] / 4.21e20, 1/3)
            rows = np.genfromtxt(folder / "new/step780_profiles.csv", delimiter=",", names=True)
            np.testing.assert_allclose(rows["ne_m3"], generated["profiles"]["density"][0])
            np.testing.assert_allclose(rows["Te_keV"], np.asarray(generated["profiles"]["temperature"])[0] / 1e3)
            self.assertFalse(out.exists())

    def test_full_calculation_uses_the_exported_pressure_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            path = folder / "wout.step780.nc"
            self.write_wout(path)
            template = folder / "base.toml"
            template.write_text('[geometry]\nn_radial = 3\n', encoding="utf-8")
            boozer = folder / "boozmn.nc"
            boozer.touch()
            out = folder / "er.png"
            seen = {}

            def solve(_path, *, config_path, **kwargs):
                seen.update(example.load_config(config_path))
                return np.asarray([.1, .5, .9]), np.asarray([1., 0., -1.]), None, np.ones(3, dtype=bool)

            with patch.object(example, "solve_initial_er_and_bootstrap_from_wout", side_effect=solve), \
                 patch.object(example, "plot_er_profile"), redirect_stdout(io.StringIO()):
                example.main([str(path), "--config", str(template), "--boozer-file", str(boozer),
                              "--wout-pressure", "--out", str(out)])
            self.assertEqual(seen["profiles"]["model"], "prescribed")
            self.assertEqual(seen["species"]["names"], ["e", "H"])
            self.assertEqual(seen["wout_pressure"]["field_abs_T"], 9.)

    def test_vacuum_negative_pressure_zero_field_and_missing_field_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wout.step780.nc"
            for values in (dict(pressure=np.zeros(5)), dict(pressure=[10., 5., -1., 1., 0.]),
                           dict(b0=0.), dict(b0=np.nan), dict(field_variable="volavgB")):
                with self.subTest(values=values):
                    self.write_wout(path, **values)
                    with self.assertRaises(ValueError):
                        example.config_with_wout_pressure({"geometry": {"n_radial": 3}}, path)

    def test_volume_average_field_is_an_explicit_alternative(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wout.step780.nc"
            self.write_wout(path, b0=4.5, field_variable="volavgB")
            result = example.config_with_wout_pressure({}, path, field_name="volavgB")
            self.assertEqual(result["wout_pressure"]["field_variable"], "volavgB")
            self.assertAlmostEqual(result["wout_pressure"]["ne_axis_m3"] / 4.21e20, .5)


if __name__ == "__main__":
    unittest.main()
