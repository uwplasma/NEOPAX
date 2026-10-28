"""Numerical terminal-objective checks, without VMEC/NTX/transport solves.

Finite differences use independent NumPy objective formulas. These checks
validate local objective derivatives and residual assembly, not the complete
transport/geometry Jacobian at the saved remote optimization point.
"""

from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from NEOPAX import optimization as opt
from NEOPAX import _reverse_ad_transport as reverse
from NEOPAX import _reverse_ad_optimization as root_reverse
from NEOPAX._constants import elementary_charge
from examples.optimization import (
    optimize_geometry_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport
    as geometry,
)
from examples.benchmarks import audit_er_transition_surrogate_objectives as surrogate_audit


TARGET = 0.514
LABELS = ("Er_transition_rho", "Er_transition_location_moment", "Er_transition_strength")


def _logsumexp(values):
    maximum = np.max(values)
    return maximum + np.log(np.sum(np.exp(values - maximum)))


def _numpy_transition(er, rho):
    """Independent implementation of the documented local objective formulas."""
    er, rho = np.asarray(er), np.asarray(rho)
    temperature, eps, beta = 2.0, 1e-6, 16.0
    left, right = er[:-1] / temperature, -er[1:] / temperature
    score = (left + right - np.hypot(left - right, eps)) / 2
    faces = (rho[:-1] + rho[1:]) / 2
    mask = (faces >= 0.05) & (faces <= 0.9)
    if not np.any(mask):
        mask[np.argmin(np.abs(faces - TARGET))] = True
    left_abs, right_abs = np.hypot(er[:-1], temperature * eps), np.hypot(er[1:], temperature * eps)
    candidates = rho[:-1] + np.diff(rho) * left_abs / (left_abs + right_abs)
    bias = -0.5 * ((candidates - TARGET) / 0.05) ** 2
    selected = bias[mask] + beta * score[mask]
    strength = (_logsumexp(selected) - _logsumexp(bias[mask])) / beta
    weights = np.exp(selected - _logsumexp(selected))
    location = np.sum(weights * candidates[mask])
    return np.asarray([location, location - TARGET, strength])


def _jax_transition(er, rho):
    state = SimpleNamespace(Er=er)
    runtime = SimpleNamespace(geometry=SimpleNamespace(rho_grid=rho))
    return jnp.stack(tuple(
        reverse.objective_scalar_by_index(
            state, runtime, index, objective_labels=LABELS,
            er_transition_rho_min=0.05, er_transition_rho_max=0.9,
            er_transition_rho_target=TARGET, er_transition_temperature_kv_m=2.0,
            er_transition_positive_part_eps=1e-6, er_transition_rho_softness=0.05,
            er_transition_softmax_beta=16.0,
        )
        for index in range(3)
    ))


def _finite_difference(function, point, steps):
    point = np.asarray(point, dtype=float)
    steps = np.broadcast_to(steps, point.shape)
    columns = []
    for index, step in enumerate(steps):
        plus, minus = point.copy(), point.copy()
        plus[index] += step
        minus[index] -= step
        columns.append((function(plus) - function(minus)) / (2 * step))
    return np.stack(columns, axis=-1)


def _numpy_bootstrap(upar, charge):
    current_weights = np.sign(charge)
    if upar.shape[0] == len(charge):
        current = np.sum(upar * current_weights[:, None], axis=0)
    else:
        current = np.sum(upar * current_weights[None, :], axis=1)
    current = current * (1e20 * float(elementary_charge) * 1e-5)
    return _logsumexp(128 * np.hypot(current, 1e-12)) / 128


class RootBootstrapDerivativeTests(unittest.TestCase):
    def test_surrogate_does_not_reproduce_fixed_cell_amplitude_targets(self):
        rows = surrogate_audit.amplitude_probe()
        np.testing.assert_allclose([row["location"] for row in rows], 0.5 + 0.02 * 26 / 36, atol=1e-14)
        np.testing.assert_allclose([row["strength"] for row in rows], [2.5, 5.0, 7.5, 10.0], atol=1e-12)
        at_one_and_a_half = rows[2]
        self.assertAlmostEqual(at_one_and_a_half["fixed_cell_cost"], 8.73, places=12)
        self.assertAlmostEqual(at_one_and_a_half["d_fixed_cell_cost_d_amplitude"], 34.92, places=12)
        self.assertLess(abs(at_one_and_a_half["d_location_d_amplitude"]), 1e-12)
        self.assertLess(abs(at_one_and_a_half["d_surrogate_cost_d_amplitude"]), 1e-12)
        # Independent chord derivatives show that the small amplitude response
        # coexists with nonzero local cotangents on both Er cells.
        er = jnp.asarray([26.0, -10.0], dtype=jnp.float64)
        rho = jnp.asarray([0.5, 0.52], dtype=jnp.float64)
        bars = jax.grad(lambda value: surrogate_audit.metrics(value, rho)[0])(er)
        np.testing.assert_allclose(bars, [0.02 * 10 / 36**2, 0.02 * 26 / 36**2], rtol=1e-12)

    def test_strength_penalty_has_negligible_gradient_after_margin_is_satisfied(self):
        satisfied = surrogate_audit.strength_penalty_probe(2.0149622433190042, 0.47338579231761335)
        self.assertLess(abs(satisfied["d_deficit_d_strength"]), 3e-13)
        self.assertLess(abs(satisfied["d_weighted_residual_d_parameter"]), 2e-12)
        self.assertLess(abs(satisfied["d_weighted_cost_d_parameter"]), 3e-24)
        unsatisfied = surrogate_audit.strength_penalty_probe(0.2, 0.47338579231761335)
        self.assertGreater(abs(unsatisfied["d_deficit_d_strength"]), 0.99)
        self.assertGreater(abs(unsatisfied["d_weighted_residual_d_parameter"]), 4.0)

    def test_sharp_face_selection_can_suppress_switching_sensitivity(self):
        rows = surrogate_audit.selection_probe()
        broad, sharp = rows[1], rows[-1]
        self.assertLess(abs(sharp["location"] - 0.46), 1e-7)
        self.assertGreater(abs(broad["d_location_d_inner_bracket_amplitude"]), 1e-2)
        self.assertLess(
            abs(sharp["d_location_d_inner_bracket_amplitude"]),
            1e-3 * abs(broad["d_location_d_inner_bracket_amplitude"]),
        )

    def test_surrogate_tracks_resolved_motion_but_grid_locks_a_sharp_crossing(self):
        # This tests actual crossing displacement, not amplitude invariance.
        rows = surrogate_audit.translation_probe()
        for row in rows:
            with self.subTest(width=row["width"]):
                self.assertAlmostEqual(row["selected_root"], row["true_root"], places=12)
                np.testing.assert_allclose(
                    row["d_selected_root_d_q"], row["analytical_chord_tracking_factor"],
                    rtol=1e-7, atol=1e-12,
                )
                np.testing.assert_allclose(
                    row["chain_rule_er_bar_dot_direction"], row["d_selected_root_d_q"],
                    rtol=1e-7, atol=1e-12,
                )
        self.assertGreater(rows[0]["d_selected_root_d_q"], 0.95)
        self.assertLess(rows[-1]["d_selected_root_d_q"], 1e-6)
        self.assertLess(
            abs(rows[-1]["d_surrogate_cost_d_q"]),
            1e-6 * abs(rows[-1]["d_true_position_cost_d_q"]),
        )

    def test_existing_root_can_starve_a_direction_that_builds_the_desired_crossing(self):
        rows = surrogate_audit.target_region_probe()
        before, after = rows[0], rows[-1]
        self.assertGreater(before["selected_root"], 0.64)
        self.assertGreater(before["strength"], 0.9)
        self.assertLess(abs(before["d_surrogate_cost_d_p"]), 1e-15)
        self.assertLess(before["d_target_sign_margin_cost_d_p"], -10.0)
        self.assertGreater(before["surrogate_cost"], 100 * after["surrogate_cost"])
        self.assertLess(abs(after["selected_root"] - 0.514), 0.01)
        near_target = next(row for row in rows if row["p"] == 14.0)
        self.assertTrue(near_target["target_bracket_exists"])
        self.assertLess(abs(near_target["target_bracket_root"] - 0.514), 1e-4)
        self.assertGreater(near_target["target_bracket_margin"], 0.9)
        self.assertGreater(near_target["selected_root"], 0.64)
        # The local sign proxy has no prescribed 26/-10 amplitude targets.
        self.assertLess(after["target_sign_margin_cost"], 1e-20)

    def test_initial_root_transition_values_and_vjp_against_numpy_fd(self):
        # Exercise the root-only dispatcher, including direct rho-grid bars.
        # These numerical checks do not call an ambipolar or transport solver.
        options = {
            "Er_transition_rho_min": 0.05,
            "Er_transition_rho_max": 0.9,
            "Er_transition_rho_target": TARGET,
            "Er_transition_temperature_kv_m": 2.0,
            "Er_transition_positive_part_eps": 1e-6,
            "Er_transition_rho_softness": 0.05,
            "Er_transition_softmax_beta": 16.0,
        }
        self.assertEqual(
            root_reverse.normalize_initial_er_root_only_objective_names(LABELS), LABELS,
        )

        def objective(er, rho):
            return root_reverse._initial_er_root_only_objective_values(
                SimpleNamespace(Er=er),
                SimpleNamespace(geometry=SimpleNamespace(rho_grid=rho)),
                LABELS, options=options,
            )

        rho = np.linspace(0.003, 0.997, 51)
        cases = (
            40 * np.tanh((0.55657 - rho) / 0.018),
            np.full(51, -10.0),
            np.zeros(51),
        )
        for er in cases:
            with self.subTest(er_range=(float(er.min()), float(er.max()))):
                values, (er_bars, rho_bars) = reverse._objective_vector_joint_vjp_rows(
                    objective, jnp.asarray(er), jnp.asarray(rho),
                )
                numerical = _finite_difference(
                    lambda x: _numpy_transition(x[:51], x[51:]),
                    np.concatenate((er, rho)),
                    np.concatenate((
                        np.full(51, 2e-9 if not np.any(er) else 1e-5),
                        np.full(51, 1e-6),
                    )),
                )
                np.testing.assert_allclose(values, _numpy_transition(er, rho), rtol=2e-12, atol=2e-12)
                np.testing.assert_allclose(
                    np.concatenate((np.asarray(er_bars), np.asarray(rho_bars)), axis=1),
                    numerical, rtol=3e-5, atol=5e-7,
                )
                self.assertGreater(float(jnp.linalg.norm(rho_bars)), 0.0)

    def test_initial_root_transition_options_reach_shared_formula(self):
        # Non-default settings catch silently ignored root_options entries.
        er = jnp.asarray([4.0, 3.0, 1.0, -2.0, -5.0])
        rho = jnp.linspace(0.1, 0.9, er.size)
        kwargs = dict(
            er_transition_rho_min=0.1, er_transition_rho_max=0.8,
            er_transition_rho_target=0.63, er_transition_temperature_kv_m=3.0,
            er_transition_positive_part_eps=0.02, er_transition_rho_softness=0.13,
            er_transition_softmax_beta=7.0,
        )
        options = {"E" + name[1:]: value for name, value in kwargs.items()}
        state = SimpleNamespace(Er=er)
        runtime = SimpleNamespace(geometry=SimpleNamespace(rho_grid=rho))
        actual = root_reverse._initial_er_root_only_objective_values(
            state, runtime, LABELS, options=options,
        )
        expected = jnp.stack(tuple(reverse.objective_scalar_by_index(
            state, runtime, index, objective_labels=LABELS, **kwargs,
        ) for index in range(len(LABELS))))
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)

    def test_transition_values_and_er_and_rho_derivatives_against_numpy_fd(self):
        rho = np.linspace(0.003, 0.997, 51)
        resolved = 40 * np.tanh((0.55657 - rho) / 0.018)
        resolved[0] = 0.0
        cases = {
            "resolved_crossing_with_axis_zero": resolved,
            "sharp_crossing": np.where(rho < 0.55657, 40.0, -12.0),
            "all_positive": np.full(51, 10.0),
            "all_negative": np.full(51, -10.0),
            "zero_profile": np.zeros(51),
        }
        derivative = jax.jit(jax.jacrev(_jax_transition, argnums=(0, 1)))
        for name, er in cases.items():
            with self.subTest(profile=name):
                actual = np.asarray(_jax_transition(jnp.asarray(er), jnp.asarray(rho)))
                np.testing.assert_allclose(actual, _numpy_transition(er, rho), rtol=2e-12, atol=2e-12)
                er_bar, rho_bar = derivative(jnp.asarray(er), jnp.asarray(rho))
                analytical = np.concatenate((np.asarray(er_bar), np.asarray(rho_bar)), axis=1)
                point = np.concatenate((er, rho))
                steps = np.concatenate((np.full(51, 2e-9 if name == "zero_profile" else 1e-5), np.full(51, 1e-6)))
                numerical = _finite_difference(lambda values: _numpy_transition(values[:51], values[51:]), point, steps)
                np.testing.assert_allclose(analytical, numerical, rtol=3e-5, atol=5e-7)
                print(f"[objective-derivatives] transition profile={name} max_abs_error={np.max(np.abs(analytical - numerical)):.3e}")

    def test_selected_location_has_a_chord_derivative_inside_one_face(self):
        rho = np.linspace(0.003, 0.997, 51)
        er = np.where(rho < 0.55657, 40.0, -12.0)
        location, _, _ = np.asarray(_jax_transition(jnp.asarray(er), jnp.asarray(rho)))
        index = np.nonzero((er[:-1] > 0) & (er[1:] < 0))[0][0]
        midpoint = (rho[index] + rho[index + 1]) / 2
        er_bar = np.asarray(jax.grad(lambda values: _jax_transition(values, jnp.asarray(rho))[1])(jnp.asarray(er)))
        self.assertGreater(abs(location - midpoint), 1e-3)
        self.assertGreater(np.linalg.norm(er_bar), 1e-6)
        self.assertLess(abs(float(er_bar @ er)), 1e-10)

    def test_grouped_terminal_vjp_routes_root_rows_and_geometry_grid_bars(self):
        rho = np.linspace(0.003, 0.997, 51)
        er = 40 * np.tanh((0.55657 - rho) / 0.018)
        er[0] = 0.0
        values, (er_bar, rho_bar) = reverse._objective_vector_joint_vjp_rows(
            _jax_transition, jnp.asarray(er), jnp.asarray(rho),
        )
        actual = np.concatenate((np.asarray(er_bar), np.asarray(rho_bar)), axis=1)
        numerical = _finite_difference(
            lambda x: _numpy_transition(x[:51], x[51:]),
            np.concatenate((er, rho)),
            np.concatenate((np.full(51, 1e-5), np.full(51, 1e-6))),
        )
        np.testing.assert_allclose(values, _numpy_transition(er, rho), rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(actual, numerical, rtol=3e-5, atol=5e-7)
        print(f"[objective-derivatives] grouped_root_vjp max_abs_error={np.max(np.abs(actual - numerical)):.3e}")

    def test_compact_bootstrap_upar_cotangent_against_direct_ad_and_numpy_fd(self):
        charge = np.asarray([-1.0, 1.0, 1.0, 2.0])
        state = SimpleNamespace(pressure=jnp.ones((4, 7), dtype=jnp.float64))
        for transpose in (False, True):
            for sign in (1.0, -1.0):
                with self.subTest(transpose=transpose, sign=sign):
                    upar = sign * np.arange(28, dtype=float).reshape((4, 7)) * 100
                    if transpose:
                        upar = upar.T
                    runtime = SimpleNamespace(species=SimpleNamespace(charge_qp=jnp.asarray(charge)))
                    value, compact_bar = reverse.bootstrap_current_softmax_abs_value_and_upar_bar(
                        state, runtime, {"Upar_neo": jnp.asarray(upar)},
                    )

                    def direct(values):
                        flux = SimpleNamespace(evaluate_momentum_corrected_fluxes=lambda _state: {"Upar_neo": values})
                        direct_runtime = SimpleNamespace(species=runtime.species, models=SimpleNamespace(flux=flux))
                        return reverse.bootstrap_current_softmax_abs_scaled(state, direct_runtime)

                    direct_bar = np.asarray(jax.grad(direct)(jnp.asarray(upar)))
                    numerical = _finite_difference(
                        lambda values: np.asarray([_numpy_bootstrap(values.reshape(upar.shape), charge)]),
                        upar.ravel(), 1e-2,
                    )[0].reshape(upar.shape)
                    np.testing.assert_allclose(float(value), _numpy_bootstrap(upar, charge), rtol=2e-13)
                    np.testing.assert_allclose(compact_bar, direct_bar, rtol=2e-13, atol=1e-15)
                    np.testing.assert_allclose(compact_bar, numerical, rtol=2e-6, atol=1e-10)
                    print(f"[objective-derivatives] bootstrap transpose={transpose} sign={sign:g} max_abs_error={np.max(np.abs(np.asarray(compact_bar) - numerical)):.3e}")

    def test_weighted_residual_chain_rule_and_parameter_scaling_against_fd(self):
        args = geometry.parser().parse_args([])
        terms = tuple(term for term in opt._normalize_initial_er_root_least_squares_terms(geometry.active_terms(args))
                      if term.objective.name in (*LABELS, "bootstrap_current_softmax_abs_scaled"))
        self.assertEqual(len(terms), 5)
        scales = np.asarray([4.21, 17.8, 10.0, 2.0, 1.0, 1.0, 0.25, 0.5])
        matrix = np.random.default_rng(321).normal(size=(5, 8)) * 0.03
        matrix[1] = matrix[0]  # Subtracting a fixed target preserves the location derivative.
        matrix[3] = matrix[2]  # The same raw strength appears as diagnostic and penalty.
        for strength, bootstrap in ((1.75, 0.565), (0.2, 0.565), (1.75, 0.05)):
            with self.subTest(strength=strength, bootstrap=bootstrap):
                baseline = np.asarray([0.55657, 0.55657 - TARGET, strength, strength, bootstrap])
                raw_result = opt.LeastSquaresResult(
                    residuals=jnp.asarray(baseline), jacobian=jnp.asarray(matrix),
                    residual_labels=tuple(term.residual_label for term in terms),
                    parameter_labels=tuple(f"parameter_{index}" for index in range(8)), objective_values={},
                )
                raw = opt.LeastSquaresEvaluation(result=raw_result, residuals=raw_result.residuals,
                                                 jacobian=raw_result.jacobian, elapsed_s=0.0)
                assembled = opt._assemble_mixed_initial_er_root_result(terms, base_evaluation=raw)
                assembled_eval = opt.LeastSquaresEvaluation(result=assembled, residuals=assembled.residuals,
                                                            jacobian=assembled.jacobian, elapsed_s=0.0)
                scaled = opt.scale_least_squares_evaluation_columns(assembled_eval, scales)

                def independent_residual(x):
                    values = baseline + matrix @ (scales * x)
                    deficit = 0.9 - values[3]
                    positive = 0.5 * (deficit + np.hypot(deficit, 1e-6))
                    return np.asarray([0.0, np.sqrt(1000) * values[1], 0.0,
                                       10 * positive, 2 * max(values[4] - 0.1, 0.0)])

                zero = np.zeros(8)
                numerical = _finite_difference(independent_residual, zero, 1e-6)
                np.testing.assert_allclose(scaled.residuals, independent_residual(zero), rtol=2e-12, atol=2e-12)
                np.testing.assert_allclose(scaled.jacobian, numerical, rtol=2e-6, atol=1e-8)
                numerical_cost = _finite_difference(
                    lambda x: np.asarray([0.5 * np.sum(independent_residual(x) ** 2)]), zero, 1e-6,
                )[0]
                actual_cost = np.asarray(scaled.jacobian).T @ np.asarray(scaled.residuals)
                np.testing.assert_allclose(actual_cost, numerical_cost, rtol=2e-6, atol=2e-7)
                print(f"[objective-derivatives] residual strength={strength:g} bootstrap={bootstrap:g} max_abs_error={np.max(np.abs(np.asarray(scaled.jacobian) - numerical)):.3e}")


if __name__ == "__main__":
    unittest.main()
