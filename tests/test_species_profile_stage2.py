"""Tiny profile-map AD/FD and mocked stage-2 wiring; no VMEC/NTX/transport solve."""

import copy
from contextlib import redirect_stdout
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

from NEOPAX import optimization as opt
from NEOPAX import _reverse_ad_optimization as root_ad
from NEOPAX import _reverse_ad_transport as transport_ad
from NEOPAX._orchestrator import _build_state
from NEOPAX._reverse_ad_parameters import (
    PROFILE_PARAMETER_ORDER, ProfileParameterSpec, configured_profile_parameter_order,
    parse_reverse_ad_parameter_spec, reverse_ad_optimization_parameter_set,
)
from NEOPAX._reverse_ad_transport import initial_state_for_parameter_vector
from NEOPAX._species_profile_parameters import species_profile_values
from examples.optimization import optimize_species_profiles_from_geometry_result as example


def fixture():
    config = tomllib.loads((example.ROOT / "examples/optimization/full_transport_database_vacuum_t2.toml").read_text())
    labels = example.independent_parameters(config)
    geometry = SimpleNamespace(r_grid=jnp.linspace(.05, .85, 6), r_grid_half=jnp.linspace(0., 1., 7),
                               rho_grid=jnp.linspace(.05, .85, 6))
    species = SimpleNamespace(number_species=4, names=config["species"]["names"],
                              charge_qp=jnp.asarray(config["species"]["charge_qp"]))
    baseline = _build_state(config, geometry, species)
    config["optimization"] = dict(species_profile_parameters=list(labels), species_profile_quasineutral=True)
    runtime = SimpleNamespace(geometry=geometry, species=species)
    values = species_profile_values(config)

    def state(p):
        return initial_state_for_parameter_vector(p, baseline_state=baseline, runtime=runtime,
            config=config, profile_cfg=config["profiles"], initial_er_root_ad="off")
    return config, labels, values, runtime, baseline, state


class SpeciesProfileTests(unittest.TestCase):
    def test_shared_layout_is_unchanged_and_species_labels_are_explicit(self):
        self.assertEqual(configured_profile_parameter_order({}), PROFILE_PARAMETER_ORDER)
        for label in ("T0[e]", "n0[D]", "density_shape_alpha[T]", "T0[2]"):
            self.assertIsInstance(parse_reverse_ad_parameter_spec(label), ProfileParameterSpec)
        with self.assertRaises(ValueError):
            ProfileParameterSpec("n_edge[D]")
        with self.assertRaises(ValueError):
            configured_profile_parameter_order({"optimization": {"species_profile_parameters": ["T0[e]", "T0[e]"]}})

    def test_expanded_seed_preserves_all_initial_profiles_and_floor_temperature(self):
        config, labels, values, runtime, baseline, state = fixture()
        self.assertEqual(len(labels), 15)
        actual = state(values)
        for field in ("density", "pressure", "temperature", "Er"):
            np.testing.assert_allclose(getattr(actual, field), getattr(baseline, field), rtol=1e-14, atol=1e-14)
        np.testing.assert_array_equal(actual.density[3], 0.)
        expected_helium_temperature = ((17.8-.7)*(1-runtime.geometry.r_grid**2)+.7)*.7
        np.testing.assert_allclose(actual.temperature[3], expected_helium_temperature)
        self.assertTrue(np.isfinite(actual.pressure).all())

    def test_temperature_dofs_are_independent_and_ion_densities_set_electron_density(self):
        config, labels, values, runtime, baseline, state = fixture()
        hot = state(values.at[labels.index("T0[e]")].add(1.))
        self.assertGreater(float(jnp.max(hot.temperature[0]-baseline.temperature[0])), 0.)
        np.testing.assert_allclose(hot.temperature[1:], baseline.temperature[1:], rtol=1e-14)
        np.testing.assert_allclose(hot.density, baseline.density, rtol=1e-14)
        dense = state(values.at[labels.index("n0[D]")].add(.5))
        np.testing.assert_allclose(dense.density[2:], baseline.density[2:], rtol=1e-14)
        np.testing.assert_allclose(dense.density[0], dense.density[1]+dense.density[2]+2*dense.density[3])
        np.testing.assert_allclose(dense.temperature, baseline.temperature, rtol=1e-14)
        with self.assertRaises(ValueError):
            example.independent_parameters(config, "n0[e]")
        with self.assertRaises(ValueError):
            example.independent_parameters(config, "T0[He]")
        with self.assertRaises(ValueError):
            example.independent_parameters(config, "n0[He]")

    def test_all_expanded_profile_columns_match_finite_differences(self):
        config, labels, values, runtime, baseline, state = fixture()

        def observables(p):
            s = state(p)
            return jnp.concatenate((s.density[1:3].reshape(-1), s.temperature[:3].reshape(-1)))

        jac = np.asarray(jax.jacrev(observables)(values))
        self.assertEqual(jac.shape, (30, 15))
        self.assertTrue(np.all(np.linalg.norm(jac, axis=0) > 1e-10))
        fd = []
        for i in range(len(values)):
            step = 1e-5 * max(1., abs(float(values[i])))
            fd.append(np.asarray((observables(values.at[i].add(step))-observables(values.at[i].add(-step)))/(2*step)))
        np.testing.assert_allclose(jac, np.stack(fd, axis=1), rtol=2e-6, atol=2e-9)

    def test_persistent_root_pullback_keeps_the_full_species_column_order(self):
        config, labels, values, runtime, baseline, state = fixture()
        current = state(values)
        bars = jax.tree_util.tree_map(lambda a: jnp.ones_like(a)[None], current)
        zero_state = jax.tree_util.tree_map(jnp.zeros_like, bars)
        support = {"geometry": {"g": jnp.array([1.])}, "database": {"d": jnp.array([2.])}}
        zero_geometry = {"g": jnp.zeros((1, 1))}
        zero_database = {"d": jnp.zeros((1, 1))}
        params = reverse_ad_optimization_parameter_set(profiles=labels, vmec_boundary=())
        with patch.object(root_ad, "initial_er_charge_flux_residual_er_derivative", return_value=jnp.ones(6)), \
             patch.object(root_ad, "compact_initial_er_state_pullback", return_value=zero_state), \
             patch.object(root_ad, "compact_initial_er_database_support_bars", return_value=zero_database), \
             patch.object(root_ad, "compact_initial_er_database_geometry_bars", return_value=zero_geometry):
            matrix, _ = root_ad._database_initial_root_to_unfolded_support_bars(
                fixed_runtime=object(), support=support, pre_root_state=current,
                er_profile=jnp.zeros(6), finite_mask=jnp.ones(6, dtype=bool), rooted_state_bars=bars,
                direct_geometry_bars=zero_geometry, direct_database_bars=zero_database,
                parameter_set=params, profile_values_arr=values,
                pre_root_state_from_profile_values=state, objective_count=1,
                profile_parameter_order=labels)
        expected = jax.grad(lambda p: jnp.sum(state(p).density)+jnp.sum(state(p).pressure))(values)
        np.testing.assert_allclose(matrix[0], expected, rtol=1e-13, atol=1e-13)
        self.assertEqual(matrix.shape, (1, 15))

    def test_profile_problem_uses_expanded_root_and_reverse_layout_and_export_map(self):
        config, labels, values, runtime, baseline, state = fixture()
        observer = Mock()
        with patch.object(opt, "_prepare_full_transport_config", side_effect=lambda c, **_: copy.deepcopy(c)), \
             patch.object(opt, "build_runtime_context", return_value=(runtime, baseline)), \
             patch.object(opt, "build_database_initial_root_experiment_stage") as root_stage, \
             patch.object(opt, "internal_realtime_geometry_transport_reverse_table_result_builder") as builder:
            problem = opt.full_transport_profile_least_squares_problem(config,
                ((opt.transport.Er_transition_location_moment, 0., 1000.),),
                profile_parameters=labels, profile_coordinate_mode="delta",
                reverse_stage_mode="database_full_transport_optimization", terminal_state_observer=observer)
        self.assertEqual(problem.parameter_labels, labels)
        np.testing.assert_array_equal(problem.x0, np.zeros(15))
        np.testing.assert_allclose(problem._scaled_to_physical(problem.x0), values)
        self.assertEqual(root_stage.call_args.kwargs["parameter_set"].profile_labels, labels)
        self.assertIs(builder.call_args.kwargs["terminal_state_observer"], observer)
        x = np.zeros(15)
        x[labels.index("n0[D]")] = .1
        x[labels.index("T0[e]")] = .2
        exported = problem.config_from_scaled_parameters(x)
        normal = _build_state(exported, runtime.geometry, runtime.species)
        ad_initial = state(problem._scaled_to_physical(x))
        for field in ("density", "pressure", "temperature", "Er"):
            np.testing.assert_allclose(getattr(normal, field), getattr(ad_initial, field), rtol=1e-13)
        json.dumps(exported)  # No traced arrays escape into saved configs.

    def test_example_runs_expanded_stage2_only_and_saves_parameters(self):
        config, labels, values, runtime, baseline, state = fixture()
        settings = example.geometry.parser().parse_args([])
        terms = tuple(t for t in example.geometry.active_terms(settings)
                      if getattr(t[0], "objective", t[0]).family == "transport")
        evaluation = SimpleNamespace(residuals=np.array([1., 2., 3.]))
        captured = {}

        class FakeProblem:
            parameter_labels = labels
            parameter_count = len(labels)
            x0 = np.zeros(len(labels))
            x_scale = np.asarray(values)
            profile_coordinate_mode = "delta"

            def _scaled_to_physical(self, x):
                return values*(1+jnp.asarray(x))

            def evaluate(self, x):
                captured.setdefault("evaluations", []).append(np.asarray(x))
                return evaluation

            def config_from_scaled_parameters(self, x):
                return config

        def build(cfg, active_terms, **kwargs):
            captured.update(config=cfg, terms=active_terms, build=kwargs)
            return FakeProblem()

        def solve(problem, x0, **kwargs):
            captured["optimizer"] = kwargs
            self.assertEqual(problem.parameter_labels, labels)
            x = np.asarray(x0).copy()
            x[0] = .01
            return SimpleNamespace(x=x, cost=.25, nfev=2, status=1, message="mock result", optimality=.1)

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            seed = folder / "input.saved"
            seed.write_text("mock accepted boundary")
            metadata = dict(seed_input=str(seed), saved_geometry_result={"directory": str(folder / "saved")})
            lower = np.array([example.stage2.combined.PROFILE_PHYSICAL_LOWER[ProfileParameterSpec(p).base_name] for p in labels])
            upper = np.array([example.stage2.combined.PROFILE_PHYSICAL_UPPER[ProfileParameterSpec(p).base_name] for p in labels])
            out = folder / "output"
            with patch.object(example, "prepare", return_value=(config, labels, terms, lower, upper, metadata)), \
                 patch.object(example.stage2.combined, "_validate_args"), \
                 patch.object(opt, "full_transport_profile_least_squares_problem", side_effect=build), \
                 patch.object(opt, "least_squares", side_effect=solve), \
                 patch.object(example.geometry, "iteration_diagnostics", return_value="mock diagnostics"), \
                 patch.object(example, "save_initial_comparison") as plot, redirect_stdout(io.StringIO()):
                example.main(["--geometry-result-dir", str(folder / "saved"), "--out-dir", str(out),
                              "--no-diagnose-memory", "--max-nfev", "2"])
            self.assertEqual(captured["build"]["profile_parameters"], labels)
            self.assertEqual(captured["build"]["profile_coordinate_mode"], "delta")
            self.assertEqual(captured["build"]["reverse_stage_mode"], "database_full_transport_optimization")
            self.assertIsNone(captured["build"]["accepted_step_limit"])
            self.assertEqual(captured["optimizer"]["bounds"][0].shape, (15,))
            self.assertIs(captured["optimizer"]["initial_evaluation"], evaluation)
            summary = json.loads((out / "optimization_summary.json").read_text())
            self.assertEqual(summary["parameter_labels"], list(labels))
            self.assertEqual(len(summary["optimized_parameters"]), 15)
            np.testing.assert_allclose(list(summary["optimized_parameters"].values())[1:], np.asarray(values)[1:])
            self.assertEqual((out / "input.QI_fixed_optimized_geometry").read_text(), seed.read_text())
            plot.assert_called_once()

    def test_plan_only_never_builds_a_runtime(self):
        config, labels, values, runtime, baseline, state = fixture()
        terms = ((opt.transport.Er_transition_location_moment, 0., 1000.),)
        with patch.object(example, "prepare", return_value=(config, labels, terms, np.zeros(15), np.ones(15)*30, {})), \
             patch.object(opt, "full_transport_profile_least_squares_problem", side_effect=AssertionError("physics forbidden")), \
             redirect_stdout(io.StringIO()) as output:
            example.main(["--geometry-result-dir", "/saved", "--plan-only"])
        self.assertEqual(len(json.loads(output.getvalue())["parameter_labels"]), 15)

    def test_direct_table_preserves_species_columns_without_vmec_fold(self):
        config, labels, values, runtime, baseline, state = fixture()
        context = transport_ad.realtime_geometry_transport_reverse_table_context(config=config,
            baseline_values=values, baseline_runtime=runtime, baseline_state=baseline,
            profile_cfg=config["profiles"], neoclassical_cfg=config["neoclassical"])
        gradients = jnp.arange(30., dtype=jnp.float64).reshape(2, 15)
        support_result = transport_ad.RealtimeGeometrySupportCotangentResult(
            objective_values=jnp.array([1., 2.]), profile_gradient_matrix=gradients,
            support_bars=(), support_component_bars_by_name={}, support_reuse_count=0,
            support_rebuild_count=0, initial_cache_pullback_used=False, initial_cache_pullback_skipped=True)
        names = ("Er_volume_average", "Er_transition_location_moment")
        params = reverse_ad_optimization_parameter_set(profiles=labels[::-1], vmec_boundary=())
        with patch.object(transport_ad, "realtime_geometry_payload_for_runtime", return_value={"kind": "ntx_exact_lij"}), \
             patch.object(transport_ad, "find_ntx_support_payload", return_value={}), \
             patch.object(transport_ad, "prepare_reverse_static_setup", return_value=SimpleNamespace(schedule_artifact=None)) as setup, \
             patch.object(transport_ad, "realtime_geometry_support_cotangents_from_parameter_vector", return_value=support_result) as reverse:
            builder = transport_ad.internal_realtime_geometry_transport_reverse_table_result_builder(
                table_context=context, geometry_context=None, objective_labels=names)
            result = builder(names[::-1], params, {})
        np.testing.assert_array_equal(setup.call_args.args[0], values)
        np.testing.assert_array_equal(reverse.call_args.kwargs["profile_values"], values)
        self.assertEqual(result.profile_parameter_labels, labels[::-1])
        np.testing.assert_array_equal(result.profile_gradient_matrix, gradients[::-1, ::-1])
        self.assertEqual(result.geometry_gradient_matrix.shape, (2, 0))

    def test_prepare_uses_accepted_geometry_metadata_and_production_weights(self):
        config, labels, values, runtime, baseline, state = fixture()
        saved = SimpleNamespace(summary={"database_grid": [25, 25, 33], "transport_max_steps": 1000},
                                baseline_input=Path("/original/input.seed"))
        metadata = {"seed_input": "/saved/input.optimized", "max_reverse_accepted_steps": 900}
        args = example.parser().parse_args(["--geometry-result-dir", "/saved"])
        with patch.object(example.stage2, "load_geometry_result", return_value=saved), \
             patch.object(example.baseline, "prepare_case", return_value=(copy.deepcopy(config), {}, metadata)) as prepare:
            actual, order, terms, lower, upper, source = example.prepare(args)
        self.assertIsNone(prepare.call_args.args[0].seed_input)
        self.assertEqual(prepare.call_args.args[0].geometry_result_dir, Path("/saved"))
        self.assertEqual(order, labels)
        self.assertEqual(args.max_reverse_accepted_steps, 900)
        normalized = opt._normalize_initial_er_root_least_squares_terms(terms)
        weights = {t.label if getattr(t, "label", None) else t.objective.name: t.weight for t in normalized}
        self.assertEqual(weights["Er_transition_location_moment"], 1000.)
        self.assertEqual(weights["Er_transition_strength_deficit"], 100.)
        self.assertEqual(weights["bootstrap_current_penalty"], 4.)
        self.assertEqual(weights["net_total_power_volume_average_mw_m3"], 10000.)
        self.assertEqual(source["seed_input"], "/saved/input.optimized")


if __name__ == "__main__":
    unittest.main()
