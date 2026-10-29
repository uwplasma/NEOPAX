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


if __name__ == "__main__":
    unittest.main()
