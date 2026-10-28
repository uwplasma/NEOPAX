"""Fixed-cell/softmax objective checks on prescribed arrays only."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from examples.benchmarks import compare_er_fixed_cell_and_softmax_objectives as audit


def numpy_reference(er):
    er = np.asarray(er)
    maximum = er.max()
    exponentials = np.exp(16 * (er - maximum))
    peak = maximum + np.log(exponentials.sum()) / 16
    costs = np.array([0.045 * (er[25] - 26)**2, 0.045 * (er[26] + 10)**2,
                      0.045 * ((er[25] - 26)**2 + (er[26] + 10)**2), 0.25 * (peak - 25)**2])
    bars = np.zeros((4, len(er)))
    bars[0, 25] = 0.09 * (er[25] - 26)
    bars[1, 26] = 0.09 * (er[26] + 10)
    bars[2] = bars[0] + bars[1]
    bars[3] = 0.5 * (peak - 25) * exponentials / exponentials.sum()
    return costs, bars


class ErFixedCellSoftmaxComparisonTests(unittest.TestCase):
    def test_production_values_and_gradients_match_independent_formulas(self):
        rho = audit.candidates.grid()
        profiles = (audit.reference_profile(), 1000 * (audit.candidates.TARGET - rho),
                    audit.reference_profile().at[36].set(40.0), jnp.zeros_like(rho))
        derivative = jax.jit(jax.jacrev(audit.reference_costs))
        direction = np.cos(7 * np.asarray(rho))
        for er in profiles:
            with self.subTest(er_range=(float(er.min()), float(er.max()))):
                expected, expected_bars = numpy_reference(er)
                actual_bars = np.asarray(derivative(er))
                np.testing.assert_allclose(audit.reference_costs(er), expected, rtol=1e-13, atol=1e-12)
                np.testing.assert_allclose(actual_bars, expected_bars, rtol=1e-13, atol=1e-13)
                step = 1e-5
                fd = (numpy_reference(er + step * direction)[0] - numpy_reference(er - step * direction)[0]) / (2 * step)
                np.testing.assert_allclose(actual_bars @ direction, fd, rtol=2e-6, atol=1e-7)

    def test_peak_gradient_follows_the_peak_while_fixed_cell_support_stays_put(self):
        for index in (10, 36):
            with self.subTest(peak_index=index):
                er = audit.reference_profile().at[index].set(40.0)
                bars = np.asarray(jax.jacrev(audit.reference_costs)(er))
                self.assertAlmostEqual(bars[3, index], 7.5, places=12)
                self.assertEqual(int(np.argmax(abs(bars[3]))), index)
                self.assertEqual(np.flatnonzero(bars[0]).tolist(), [25])
                self.assertEqual(np.flatnonzero(bars[1]).tolist(), [26])
                self.assertEqual(bars[2, index], 0.0)

    def test_softmax_observable_has_exact_unit_response_to_a_common_offset(self):
        er = audit.reference_profile().at[36].set(40.0)
        observables = audit.reference_observables(er)
        np.testing.assert_allclose(audit.reference_observables(er + 3.5), observables + 3.5, atol=1e-13)
        derivatives = jax.jacfwd(lambda offset: audit.reference_observables(er + offset))(0.0)
        np.testing.assert_allclose(derivatives, 1.0, atol=1e-14)

    def test_softmax_offset_fit_can_succeed_while_opposing_cell_targets_cannot(self):
        fits = {row["objective"]: row for row in audit.offset_only_fits()}
        peak_fit = fits["softmax_Er_target"]
        self.assertLess(peak_fit["raw_objective_cost"], 1e-24)
        self.assertAlmostEqual(peak_fit["softmax_Er"], 25.0, places=12)
        self.assertEqual(peak_fit["downcrossing_count"], 0)
        pair_fit = fits["Er_left_right_targets"]
        self.assertAlmostEqual(pair_fit["Er_left"], 8.0, places=8)
        self.assertAlmostEqual(pair_fit["Er_right"], 8.0, places=8)
        self.assertAlmostEqual(pair_fit["raw_objective_cost"], 29.16, places=8)
        er = audit.reference_profile()
        derivative = jax.jacfwd(lambda offset: audit.reference_costs(er + offset))(0.0)
        self.assertLess(float(derivative[0]), 0.0)
        self.assertGreater(float(derivative[1]), 0.0)

    def test_cell_pair_has_creation_gradient_that_current_selector_lacks(self):
        er, rho = audit.reference_profile(), audit.candidates.grid()
        derivative = np.asarray(jax.jacfwd(lambda p: audit.all_losses(er.at[26].add(-p), rho))(0.0))
        self.assertLess(abs(derivative[0]), 1e-15)
        self.assertLess(derivative[audit.NAMES.index("Er_left_right_targets")], -1.0)
        self.assertGreater(derivative[audit.NAMES.index("softmax_Er_target")], 0.0)
        self.assertLess(derivative[audit.NAMES.index("target_zero_regional")], -0.5)

    def test_cell_targets_can_be_met_while_a_wrong_outer_pocket_remains(self):
        er = audit.reference_profile().at[25].set(26.0).at[26].set(-10.0)
        costs = np.asarray(audit.all_losses(er, audit.candidates.grid()))
        self.assertEqual(costs[audit.NAMES.index("Er_left_right_targets")], 0.0)
        self.assertGreater(costs[audit.NAMES.index("softmax_Er_target")], 0.24)
        self.assertGreater(costs[audit.NAMES.index("target_zero_regional")], 1.0)
        self.assertEqual(audit.candidates.profile_metrics(er, audit.candidates.grid())["downcrossing_count"], 2)

    def test_joint_peak_and_left_targets_have_a_nonzero_cost_lower_bound(self):
        compatibility = audit.joint_target_compatibility()
        self.assertFalse(compatibility["simultaneous_zero_possible"])
        er = jnp.full_like(audit.reference_profile(), -10.0).at[25].set(compatibility["shared_peak_at_bound"])
        costs = audit.reference_costs(er)
        self.assertAlmostEqual(float(costs[2] + costs[3]), 0.03813559322033898, places=12)
        self.assertAlmostEqual(float(costs[2] + costs[3]), compatibility["joint_cost_lower_bound"], places=12)
        derivative = jax.grad(lambda value: jnp.sum(audit.reference_costs(er.at[25].set(value))[2:]))
        self.assertLess(abs(float(derivative(compatibility["shared_peak_at_bound"]))), 1e-12)


if __name__ == "__main__":
    unittest.main()
