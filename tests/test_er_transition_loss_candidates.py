"""Candidate loss behavior on short prescribed arrays; no plasma solves."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from examples.benchmarks import compare_er_transition_loss_candidates as candidates


def numpy_losses(er, rho):
    """Independent scalar formulas for the audit candidates and reference."""
    er, rho = np.asarray(er), np.asarray(rho)
    target, margin, scale = candidates.TARGET, 0.9, 2.0
    positive = lambda value: 0.5 * (value + np.hypot(value, 1e-6))
    inner, outer = er[:-1] / scale, -er[1:] / scale
    score = 0.5 * (inner + outer - np.hypot(inner - outer, 1e-6))
    magnitude = np.hypot(er, 2e-6)
    location = rho[:-1] + np.diff(rho) * magnitude[:-1] / (magnitude[:-1] + magnitude[1:])
    bias = -0.5 * ((location - target) / 0.05)**2
    faces = 0.5 * (rho[:-1] + rho[1:])
    mask = (faces >= 0.05) & (faces <= 0.9)
    if not mask.any():
        mask[np.argmin(abs(faces - target))] = True

    def logsumexp(values):
        maximum = np.max(values)
        return maximum + np.log(np.exp(values - maximum).sum())

    def selected(scores):
        logits = bias[mask] + 16 * scores[mask]
        weights = np.exp(logits - logsumexp(logits))
        return weights @ location[mask]

    strength = (logsumexp((bias + 16 * score)[mask]) - logsumexp(bias[mask])) / 16
    penalty = 50 * positive(margin - strength)**2
    current = 500 * (selected(score) - target)**2 + penalty
    cap = margin - 0.25 * np.logaddexp(0.0, (margin - score) / 0.25)
    capped = 500 * (selected(cap) - target)**2 + penalty
    left = np.clip(np.searchsorted(rho, target, side="left") - 1, 0, rho.size - 2)
    right = np.clip(np.searchsorted(rho, target, side="right"), 1, rho.size - 1)
    local = 0.5 * np.mean(positive(margin - np.array([er[left], -er[right]]) / scale)**2)
    zero = 0.5 * (np.interp(target, rho, er) / scale)**2
    nodes = (rho >= 0.05) & (rho <= 0.9)
    half_cell = 0.5 * np.mean(np.diff(rho))
    signed_distance = (target - rho) / np.hypot(target - rho, half_cell)
    regional = 0.5 * np.mean(positive(margin * signed_distance[nodes]**2 - signed_distance[nodes] * er[nodes] / scale)**2)
    return np.array([current, capped, local, zero + local, zero + regional])


class ErTransitionLossCandidatesTests(unittest.TestCase):
    def test_values_and_directional_gradients_match_independent_numpy_fd(self):
        rho = candidates.grid()
        direction = np.cos(np.asarray(rho) * 7.0)
        profiles = (
            10 * np.tanh((0.65 - np.asarray(rho)) / 0.01),
            500 * (candidates.TARGET - np.asarray(rho)),
            -10 * np.tanh((candidates.TARGET - np.asarray(rho)) / 0.02),
            np.zeros(51),
        )
        derivative = jax.jit(jax.jacrev(candidates.losses))
        for er in profiles:
            with self.subTest(er_range=(float(er.min()), float(er.max()))):
                actual = np.asarray(candidates.losses(jnp.asarray(er), rho))
                np.testing.assert_allclose(actual, numpy_losses(er, rho), rtol=1e-12, atol=1e-12)
                bars = np.asarray(derivative(jnp.asarray(er), rho))
                self.assertTrue(np.isfinite(bars).all())
                step = 1e-5 if np.any(er) else 1e-8
                fd = (numpy_losses(er + step * direction, rho) - numpy_losses(er - step * direction, rho)) / (2 * step)
                np.testing.assert_allclose(bars @ direction, fd, rtol=3e-5, atol=2e-6)

    def test_reference_calibration_is_frozen_and_equalizes_one_profile(self):
        rho = candidates.grid()
        er = 10 * jnp.tanh((0.65 - rho) / 0.01)
        values = np.asarray(candidates.losses(er, rho)) * candidates.calibration()
        np.testing.assert_allclose(values, values[0], rtol=1e-13)

    def test_target_local_terms_guide_creation_before_a_new_root_wins(self):
        rho = candidates.grid()
        er = 10 * jnp.tanh((0.65 - rho) / 0.01)
        derivatives = jax.jacfwd(lambda p: candidates.losses(er.at[26].add(-p), rho))(0.0)
        self.assertLess(abs(float(derivatives[0])), 1e-15)
        self.assertLess(abs(float(derivatives[1])), 1e-15)
        self.assertTrue(np.all(np.asarray(derivatives[2:]) < -0.5))

    def test_regional_term_sees_extra_roots_after_local_terms_are_satisfied(self):
        rho = candidates.grid()
        remote = 10 * jnp.tanh((0.65 - rho) / 0.01)
        er = remote.at[26].add(-14.0)
        direction = jnp.where((rho > rho[26]) & (rho < 0.65), -2 * jnp.maximum(remote, 0), 0)
        values = np.asarray(candidates.losses(er, rho))
        derivatives = np.asarray(jax.jacfwd(lambda p: candidates.losses(er + p * direction, rho))(0.0))
        self.assertEqual(candidates.profile_metrics(er, rho)["downcrossing_count"], 2)
        self.assertLess(values[2], 1e-20)
        self.assertEqual(derivatives[2], 0.0)
        self.assertEqual(derivatives[3], 0.0)
        self.assertLess(derivatives[4], -1.0)

    def test_average_of_distant_crossings_can_falsely_match_the_target(self):
        rho = jnp.asarray([candidates.TARGET - 0.13, candidates.TARGET - 0.11,
                           candidates.TARGET + 0.11, candidates.TARGET + 0.13])
        er = jnp.asarray([10.0, -10.0, 10.0, -10.0])
        values = np.asarray(candidates.losses(er, rho))
        self.assertTrue(np.all(values[:2] < 1e-20))
        self.assertTrue(np.all(values[2:] > 1.0))
        self.assertAlmostEqual(candidates.profile_metrics(er, rho)["nearest_downcrossing_error"], 0.12)

    def test_local_margin_plateau_does_not_pin_the_zero_radius(self):
        rho = candidates.grid()
        er = 500 * (0.506 - rho)
        values = np.asarray(candidates.losses(er, rho))
        self.assertLess(values[2], 1e-20)
        self.assertGreater(values[3], 1.0)
        derivative = jax.jacfwd(lambda q: candidates.losses(500 * (q - rho), rho))(0.506)
        self.assertLess(float(derivative[3]), -1.0)

    def test_correct_crossing_has_no_above_margin_amplitude_target(self):
        rho = candidates.grid()
        er = 1000 * (candidates.TARGET - rho)
        values = np.asarray(candidates.losses(er, rho))
        derivatives = np.asarray(jax.jacfwd(lambda a: candidates.losses(a * er, rho))(1.0))
        self.assertTrue(np.all(values[2:] < 1e-18))
        self.assertTrue(np.all(np.abs(derivatives[2:]) < 1e-18))

    def test_zero_at_a_grid_node_does_not_conflict_with_sign_margins(self):
        rho = candidates.grid().at[25].set(candidates.TARGET)
        er = 1000 * (candidates.TARGET - rho)
        values = np.asarray(candidates.losses(er, rho))
        self.assertTrue(np.all(values[2:] < 1e-12))
        self.assertTrue(np.isfinite(np.asarray(jax.jacrev(candidates.losses)(er, rho))).all())
        physical = candidates.profile_metrics(er, rho)
        self.assertEqual(physical["downcrossing_count"], 1)
        self.assertEqual(physical["nearest_downcrossing_error"], 0.0)
        location, strength = candidates.audit.metrics(er, rho)
        self.assertAlmostEqual(float(location), candidates.TARGET, places=7)
        self.assertLess(float(strength), 0.0)
        self.assertGreater(values[0], 40.0)

    def test_current_bracket_strength_can_push_a_crossing_away_from_target(self):
        rho = candidates.grid()
        derivative = np.asarray(jax.jacfwd(lambda q: candidates.losses(10 * jnp.tanh((q - rho) / 0.02), rho))(0.58))
        # q already exceeds the target. Negative dC/dq encourages increasing
        # q further because the sampled bracket margin dominates placement.
        self.assertLess(derivative[0], -1000)
        self.assertGreater(derivative[4], 1.0)

    def test_requested_zero_is_interpolated_between_grid_nodes(self):
        rho = candidates.grid()
        self.assertFalse(bool(jnp.any(rho == candidates.TARGET)))
        er = candidates.profile_with_free_core(candidates.TARGET, -0.2)
        self.assertGreater(float(er[25]), 0.0)
        self.assertLess(float(er[26]), 0.0)
        self.assertGreater(abs(float(er[25])), 1.0)
        self.assertGreater(abs(float(er[26])), 1.0)
        # Neither endpoint is near zero, but their interpolated value is.
        at_target = np.interp(candidates.TARGET, np.asarray(rho), np.asarray(er))
        self.assertLess(abs(at_target), 1e-13)
        value = candidates.target_zero_region_cost(er, rho, rho_min=0.30, rho_max=0.9)
        self.assertLess(float(value), 1e-20)

    def test_free_core_zeros_and_small_negative_values_do_not_change_annular_cost(self):
        rho = candidates.grid()
        root = candidates.TARGET + 0.008  # Preserve a nonzero placement error.
        values = []
        for amplitude in (-0.2, 0.0, 0.2):
            er = candidates.profile_with_free_core(root, amplitude)
            objective = lambda field: candidates.target_zero_region_cost(field, rho, rho_min=0.30, rho_max=0.9)
            value, bars = jax.value_and_grad(objective)(er)
            values.append(float(value))
            self.assertGreater(float(value), 1.0)
            np.testing.assert_array_equal(np.asarray(bars)[np.asarray(rho) < 0.28], 0.0)
        np.testing.assert_array_equal(values, np.full(3, values[0]))

    def test_broad_inner_positive_margin_penalizes_an_allowed_weak_core(self):
        rho = candidates.grid()
        er = candidates.profile_with_free_core(candidates.TARGET, -0.2)
        broad = candidates.target_zero_region_cost(er, rho, rho_min=0.05, rho_max=0.9)
        annular = candidates.target_zero_region_cost(er, rho, rho_min=0.30, rho_max=0.9)
        self.assertGreater(float(broad), 0.01)
        self.assertLess(float(annular), 1e-20)
        # Excluding the core must not also exempt the intended positive annulus.
        bad_annulus = er.at[20].set(-0.2)
        value, bars = jax.value_and_grad(lambda field: candidates.target_zero_region_cost(
            field, rho, rho_min=0.30, rho_max=0.9,
        ))(bad_annulus)
        self.assertGreater(float(value), 0.001)
        self.assertLess(float(bars[20]), -0.001)

    def test_annular_loss_moves_the_target_crossing_even_when_the_core_has_zeros(self):
        rho = candidates.grid()
        objective = lambda q: candidates.target_zero_region_cost(
            candidates.profile_with_free_core(q, 0.0), rho, rho_min=0.30, rho_max=0.9,
        )
        inside = float(jax.grad(objective)(candidates.TARGET - 0.008))
        outside = float(jax.grad(objective)(candidates.TARGET + 0.008))
        self.assertLess(inside, 0.0)
        self.assertGreater(outside, 0.0)
        for root, ad in ((candidates.TARGET - 0.008, inside), (candidates.TARGET + 0.008, outside)):
            step = 1e-6
            fd = (float(objective(root + step)) - float(objective(root - step))) / (2 * step)
            np.testing.assert_allclose(ad, fd, rtol=2e-6, atol=1e-7)
        self.assertLess(abs(float(jax.grad(objective)(candidates.TARGET))), 1e-10)

    def test_annular_cost_direction_matches_independent_numpy_fd(self):
        rho = np.asarray(candidates.grid())
        er = np.asarray(candidates.profile_with_free_core(candidates.TARGET + 0.008, -0.2)).copy()
        er[20] = -0.2
        direction = np.cos(9.0 * rho)

        def independent(field):
            distance = candidates.TARGET - rho
            signed = distance / np.hypot(distance, 0.5 * np.mean(np.diff(rho)))
            mask = (rho >= 0.30) & (rho <= 0.9)
            deficit = 0.9 * signed**2 - signed * field / 2.0
            positive = 0.5 * (deficit + np.hypot(deficit, 1e-6))
            return 0.5 * (np.interp(candidates.TARGET, rho, field) / 2.0)**2 + 0.5 * np.mean(positive[mask]**2)

        value, bars = jax.value_and_grad(lambda field: candidates.target_zero_region_cost(
            field, jnp.asarray(rho), rho_min=0.30, rho_max=0.9,
        ))(jnp.asarray(er))
        step = 1e-5
        fd = (independent(er + step * direction) - independent(er - step * direction)) / (2 * step)
        np.testing.assert_allclose(float(value), independent(er), rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(float(np.asarray(bars) @ direction), fd, rtol=2e-6, atol=1e-8)

    def test_one_root_weight_scales_the_whole_block_and_its_gradient(self):
        rho = candidates.grid()
        er = candidates.profile_with_free_core(candidates.TARGET + 0.008, -0.2)
        objective = lambda field, weight: candidates.target_zero_region_cost(
            field, rho, rho_min=0.30, rho_max=0.9, root_weight=weight,
        )
        base_value, base_bars = jax.value_and_grad(objective)(er, 1.0)
        self.assertGreater(float(base_value), 1.0)
        for weight in (0.0, 0.1, 10.0):
            value, bars = jax.value_and_grad(objective)(er, weight)
            self.assertTrue(np.isfinite(np.asarray(bars)).all())
            np.testing.assert_allclose(value, weight * base_value, rtol=1e-13, atol=1e-13)
            np.testing.assert_allclose(bars, weight * base_bars, rtol=1e-13, atol=1e-13)


if __name__ == "__main__":
    unittest.main()
