"""Small closure/projection/AD checks; no transport, VMEC or NTX solves."""

import dataclasses
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from NEOPAX._boundary_conditions import BoundaryConditionModel
from NEOPAX._orchestrator import (
    _build_state,
    _normalize_solver_config,
    prepare_transport_solver_components,
)
from NEOPAX._state import TransportState
from NEOPAX._temperature_constraints import (
    helium_temperature_match_indices,
    matched_helium_temperature_profile,
    match_helium_temperature_boundary,
)
from NEOPAX._transport_equations import ComposedEquationSystem
from NEOPAX._transport_flux_models import (
    build_evaluated_transport_state,
    build_ntss_like_face_transport_state,
)
from NEOPAX._transport_solvers import (
    _apply_quasi_neutrality_output,
    _make_solver_state_transform,
    _pack_transport_state_arrays,
    _project_packed_transport_state_arrays,
    _project_state_to_quasi_neutrality,
    _unpack_transport_state_arrays,
)


def species(names=("e", "D", "T", "He")):
    charges = {"e": -1., "D": 1., "T": 1., "He": 2.}
    return SimpleNamespace(
        names=names,
        number_species=len(names),
        charge_qp=jnp.asarray([charges[n] for n in names]),
        ion_indices=tuple(i for i, n in enumerate(names) if n != "e"),
        species_idx={n: i for i, n in enumerate(names)},
    )


def config(enabled=True, names=("e", "D", "T", "He")):
    return {
        "species": {"names": list(names), "n_species": len(names)},
        "equations": {
            "match_THe": enabled,
            "toggle_temperature": [n != "He" for n in names],
            "toggle_density": [n == "He" for n in names],
            "toggle_Er": False,
        },
    }


def state():
    n = jnp.asarray([[1.2, 1.4, 1.6], [.5, .6, .7], [.5, .6, .7], [.1, .1, .1]])
    temperature = jnp.asarray([[5., 6., 7.], [10., 20., 30.], [8., 9., 10.], [4., 4., 4.]])
    return TransportState(density=n, pressure=n * temperature, Er=jnp.zeros(3))


def geometry():
    return SimpleNamespace(
        r_grid=jnp.asarray([.1, .3, .5]),
        r_grid_half=jnp.asarray([0., .2, .4, .6]),
        dr=.2,
    )


class HeliumTemperatureMatchTests(unittest.TestCase):
    def system(self, enabled=True, initial=None):
        initial = state() if initial is None else initial
        cfg = config(enabled)
        return ComposedEquationSystem(
            equations=(), species=species(), density_floor=1e-3,
            temperature_active_mask=jnp.asarray(cfg["equations"]["toggle_temperature"]),
            fixed_temperature_profile=matched_helium_temperature_profile(
                initial.temperature,
                helium_temperature_match_indices(cfg, species().names),
            ),
        )

    def test_false_and_omitted_keep_original_projection_exactly(self):
        initial = state()
        current = dataclasses.replace(initial, pressure=2 * initial.pressure)
        for flag in (False, None):
            cfg = config(False)
            if flag is None:
                del cfg["equations"]["match_THe"]
            indices = helium_temperature_match_indices(cfg, species().names)
            self.assertIsNone(indices)
            original_profile = initial.temperature
            profile = matched_helium_temperature_profile(original_profile, indices)
            self.assertIs(profile, original_profile)
            system = dataclasses.replace(self.system(False), fixed_temperature_profile=profile)
            result, _ = system._prepare_working_state(current)
            expected = current.pressure.at[3].set(current.density[3] * initial.temperature[3])
            np.testing.assert_array_equal(result.pressure, expected)
            np.testing.assert_array_equal(result.temperature[3], initial.temperature[3])

    def test_live_match_follows_multiple_deuterium_temperatures(self):
        initial = state()
        system = self.system()
        operation = jax.jit(lambda s: system._prepare_working_state(s)[0])
        for scale in (1., 1.5, 2.):
            current = dataclasses.replace(initial, pressure=initial.pressure.at[1].multiply(scale))
            actual = operation(current)
            np.testing.assert_allclose(actual.temperature[3], current.temperature[1], rtol=1e-14)
            np.testing.assert_array_equal(actual.pressure[:3], current.pressure[:3])

    def test_false_boundary_projection_returns_same_object(self):
        bc = BoundaryConditionModel(dr=.2, right_value=jnp.arange(4.))
        self.assertIs(match_helium_temperature_boundary(bc, None), bc)

    def test_match_inherits_species_boundary_and_face_gradients(self):
        bc = BoundaryConditionModel(
            dr=.2, left_type="neumann", right_type="dirichlet",
            left_gradient=jnp.asarray([0., 1., 2., 100.]),
            right_value=jnp.asarray([1., 2., 3., 100.]),
        )
        matched_bc = match_helium_temperature_boundary(bc, (3, 1))
        prepared, _ = self.system()._prepare_working_state(state())
        evaluated = build_evaluated_transport_state(
            prepared, geometry(), bc_temperature=matched_bc, density_floor=1e-3,
        )
        ntss_face = build_ntss_like_face_transport_state(
            prepared, geometry(), bc_temperature=matched_bc, density_floor=1e-3,
        )
        for values in (
            evaluated.center.temperature, evaluated.face.temperature,
            evaluated.temperature_grad_center, evaluated.temperature_grad_face,
            ntss_face.temperature,
        ):
            np.testing.assert_allclose(values[3], values[1], rtol=1e-13, atol=1e-13)
        np.testing.assert_array_equal(bc.right_value, [1., 2., 3., 100.])

    def test_species_order_and_configuration_validation(self):
        names = ("He", "T", "e", "D")
        self.assertEqual(helium_temperature_match_indices(config(names=names), names), (0, 3))
        for cfg, names in (
            (config(True, ("e", "D", "T")), ("e", "D", "T")),
            ({"equations": {"match_THe": "false"}}, species().names),
            ({"equations": {"match_THe": True}}, species().names),
        ):
            with self.subTest(cfg=cfg), self.assertRaisesRegex(ValueError, "match_THe"):
                helium_temperature_match_indices(cfg, names)

    def test_species_floor_inheritance_only_when_enabled(self):
        for enabled in (False, True):
            cfg = config(enabled)
            cfg["transport_solver"] = {"temperature_floor": [.1, .2, .3, 10.]}
            normalized = _normalize_solver_config(cfg)
            np.testing.assert_allclose(
                normalized["temperature_floor"], [.1, .2, .3, .2 if enabled else 10.]
            )
            self.assertEqual(cfg["transport_solver"]["temperature_floor"][-1], 10.)

    def test_initial_profiles_respect_match_without_a_solve(self):
        for enabled in (False, True):
            cfg = config(enabled)
            cfg["profiles"] = {
                "model": "standard_analytical", "n0": 4., "n_edge": .5,
                "T0": 10., "T_edge": 1., "c_density": [1., .5, .5, 0.],
                "c_temperature": [1., 1., 1., .2],
            }
            initial = _build_state(cfg, geometry(), species())
            expected_ratio = 1. if enabled else .2
            np.testing.assert_allclose(initial.temperature[3], expected_ratio * initial.temperature[1])

    def test_orchestrator_wires_the_constraint_and_boundary_without_a_solve(self):
        cfg = config()
        cfg["boundary"] = {
            "temperature": {
                "right": {"type": "dirichlet", "value": {"default": 99., "D": 2.}}
            }
        }
        runtime = SimpleNamespace(
            geometry=geometry(), species=species(),
            models=SimpleNamespace(flux=None, source={}),
            solver_parameters=_normalize_solver_config(cfg),
        )
        with patch("NEOPAX._transport_equations.build_equation_system", return_value=()), patch(
            "NEOPAX._transport_solvers.build_time_solver", return_value=object()
        ):
            prepared = prepare_transport_solver_components(cfg, runtime, state())
        system = prepared["equation_system"]
        actual, _ = system._prepare_working_state(state())
        np.testing.assert_allclose(actual.temperature[3], actual.temperature[1])
        np.testing.assert_array_equal(prepared["bc"]["temperature"].right_value, [99., 2., 99., 2.])

    def test_zero_helium_initialization_matches_with_benchmark_density_floor(self):
        cfg = config()
        cfg["transport_solver"] = {"density_floor": 1e-3}
        cfg["profiles"] = {
            "model": "standard_analytical", "n0": 4., "n_edge": .5,
            "T0": 10., "T_edge": 1., "c_density": [1., .5, .5, 0.],
            "c_temperature": [1., 1., 1., .2],
        }
        initial = _build_state(cfg, geometry(), species())
        np.testing.assert_array_equal(initial.density[3], 0.)
        np.testing.assert_allclose(initial.temperature[3], initial.temperature[1])
        prepared, _ = self.system(initial=initial)._prepare_working_state(initial)
        np.testing.assert_allclose(prepared.temperature[3], prepared.temperature[1])

    def test_packed_and_output_projections_share_the_live_constraint(self):
        initial = state()
        system = self.system()
        kwargs = dict(
            temperature_active_mask=system.temperature_active_mask,
            fixed_temperature_profile=system.fixed_temperature_profile,
            density_floor=system.density_floor,
            temperature_floor=.1,
        )
        packed = _pack_transport_state_arrays(initial, species())
        projected = jax.jit(lambda x: _project_packed_transport_state_arrays(
            x, initial, species(), **kwargs
        ))(packed)
        restored = _unpack_transport_state_arrays(projected, initial, species(), **kwargs)
        output = _project_state_to_quasi_neutrality(initial, species(), **kwargs)
        for result in (restored, output):
            np.testing.assert_allclose(result.temperature[3], result.temperature[1])
        saved = jax.tree_util.tree_map(lambda x: jnp.stack([x, 2 * x]), initial)
        saved_out = _apply_quasi_neutrality_output(saved, species(), initial, **kwargs)
        np.testing.assert_allclose(saved_out.temperature[:, 3], saved_out.temperature[:, 1])

    def test_reverse_pullback_routes_helium_pressure_to_deuterium(self):
        initial = state()
        system = self.system()
        bar = TransportState(
            density=jnp.zeros_like(initial.density),
            pressure=jnp.zeros_like(initial.pressure).at[3].set(1.), Er=jnp.zeros(3),
        )
        result = jax.jit(system._prepare_working_state_pullback)(initial, bar)
        expected_p = np.zeros_like(initial.pressure)
        expected_p[1] = initial.density[3] / initial.density[1]
        np.testing.assert_allclose(result.pressure, expected_p, atol=1e-14)
        np.testing.assert_allclose(result.density[3], initial.temperature[1])
        np.testing.assert_allclose(
            result.density[1], -initial.density[3] * initial.pressure[1] / initial.density[1]**2
        )

    def test_checkpoint_transform_ad_agrees_with_finite_difference(self):
        initial = state()
        system = self.system()
        flat, unpack, _, _, project = _make_solver_state_transform(
            initial, species(), temperature_active_mask=system.temperature_active_mask,
            fixed_temperature_profile=system.fixed_temperature_profile, density_floor=1e-3,
        )
        objective = jax.jit(lambda y: jnp.sum(unpack(project(y)).pressure[3]))
        direction = jnp.linspace(.01, .03, flat.size)
        observed_ad = jnp.vdot(jax.grad(objective)(flat), direction)
        delta = 1e-5
        observed_fd = (objective(flat + delta * direction) - objective(flat - delta * direction)) / (2 * delta)
        np.testing.assert_allclose(observed_ad, observed_fd, rtol=1e-7, atol=1e-8)

    def test_zero_helium_and_active_donor_floor_have_finite_pullbacks(self):
        initial = state()
        initial = dataclasses.replace(
            initial,
            density=initial.density.at[3].set(0.),
            pressure=initial.pressure.at[1].set(-1.).at[3].set(0.),
        )
        system = dataclasses.replace(self.system(initial=initial), temperature_floor=.5)
        result, pullback = jax.vjp(lambda s: system._prepare_working_state(s)[0], initial)
        np.testing.assert_allclose(result.temperature[3], .5)
        bar = jax.tree_util.tree_map(jnp.ones_like, result)
        actual = jax.jit(pullback)(bar)[0]
        for leaf in jax.tree_util.tree_leaves(actual):
            self.assertTrue(np.all(np.isfinite(leaf)))

    def test_original_whe_examples_explicitly_remain_disabled(self):
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        root = Path(__file__).resolve().parents[1] / "examples" / "Solve_Transport_Equations"
        files = [path for path in root.glob("*wHe*.toml") if "_match_THe" not in path.stem]
        self.assertEqual(len(files), 5)
        for path in files:
            with self.subTest(path=path):
                cfg = tomllib.loads(path.read_text())
                self.assertIs(cfg["equations"]["match_THe"], False)
                self.assertIs(cfg["equations"]["match_workHe"], False)
                self.assertIsNone(helium_temperature_match_indices(cfg, cfg["species"]["names"]))

    def test_matched_example_only_changes_ntss_matching_and_output_directories(self):
        import copy
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        root = Path(__file__).resolve().parents[1] / "examples" / "Solve_Transport_Equations"
        base = tomllib.loads((root / "Solve_Transport_equations_wHe_radau.toml").read_text())
        matched = tomllib.loads((root / "Solve_Transport_equations_wHe_radau_match_THe.toml").read_text())
        expected = copy.deepcopy(base)
        expected["equations"]["match_THe"] = True
        expected["equations"]["match_workHe"] = True
        expected["sources"]["parameters"]["bremsstrahlung_radiation"]["coefficient_mode"] = "ntssfusion"
        expected["transport_solver"]["Er_diffusion_mode"] = "ntssfusion"
        expected["transport_solver"]["temperature_convection_temperature_mode"] = "face"
        expected["ambipolarity"]["er_ambipolar_output_dir"] = "./outputs/my_ambipolarity_run_scaled_flux_match_THe_workHe_brems_er_ntss_face_T"
        expected["transport_output"]["transport_output_dir"] = "./outputs/transport_solver_wHe_radau_ntss_match_THe_workHe_brems_er_ntss_face_T"
        self.assertEqual(matched, expected)
        self.assertTrue(matched["transport_output"]["transport_plot"])
        self.assertEqual(helium_temperature_match_indices(matched, matched["species"]["names"]), (3, 1))


if __name__ == "__main__":
    unittest.main()
