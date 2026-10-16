"""Small floor-map regressions; no VMEC, NTX, or transport solve is run."""

import dataclasses
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from NEOPAX._state import TransportState, apply_transport_temperature_floor
from NEOPAX._transport_equations import ComposedEquationSystem


class TemperatureFloorPullbackTests(unittest.TestCase):
    def test_inactive_floor_preserves_large_pressure_bars_without_density_leakage(self):
        density = jnp.asarray([[1e-6, 2.7e-6, 1.]], dtype=jnp.float64)
        pressure = jnp.asarray([[12.5e-6, 3.375e-5, 12.5]], dtype=jnp.float64)
        state = TransportState(density=density, pressure=pressure, Er=jnp.zeros(3))

        def output(n, p):
            return apply_transport_temperature_floor(
                dataclasses.replace(state, density=n, pressure=p),
                temperature_floor=1e-6, density_floor=1e-6,
            ).pressure

        def pullback(n, p, bar):
            return jax.vjp(output, n, p)[1](bar)

        for name, operation in (("eager", pullback), ("jit", jax.jit(pullback))):
            for magnitude in (1., 1e150, 1e300, 3e307):
                with self.subTest(mode=name, pressure_bar=magnitude):
                    bars = jnp.full_like(pressure, magnitude)
                    n_bar, p_bar = jax.device_get(operation(density, pressure, bars))
                    self.assertTrue(np.all(np.isfinite(n_bar)), repr(n_bar))
                    self.assertTrue(np.all(np.isfinite(p_bar)), repr(p_bar))
                    # Above the temperature floor the map is exactly p -> p.
                    # A density cotangent here is numerical cancellation error.
                    np.testing.assert_array_equal(n_bar, np.zeros_like(n_bar))
                    np.testing.assert_array_equal(p_bar, np.asarray(bars))

    def test_species_floors_and_tie_have_the_expected_derivatives(self):
        density = jnp.asarray([[0., 2e-6], [.1, .2]], dtype=jnp.float64)
        pressure = jnp.asarray([[-1e-5, 1e-6], [.05, .6]], dtype=jnp.float64)
        state = TransportState(density=density, pressure=pressure, Er=jnp.zeros(2))

        def output(n, p):
            return apply_transport_temperature_floor(
                dataclasses.replace(state, density=n, pressure=p),
                temperature_floor=jnp.asarray([.5, 2.]),
                density_floor=jnp.asarray([1e-6, .15]),
            ).pressure

        value, pullback = jax.vjp(output, density, pressure)
        n_bar, p_bar = pullback(jnp.ones_like(pressure))
        np.testing.assert_allclose(value, [[.5e-6, 1e-6], [.3, .6]], rtol=1e-15)
        np.testing.assert_allclose(n_bar, [[0., .25], [0., 0.]], atol=0, rtol=1e-15)
        np.testing.assert_allclose(p_bar, [[0., .5], [0., 1.]], atol=0, rtol=1e-15)

    def test_composed_rhs_preparation_pullback_with_fixed_helium_temperature(self):
        density = jnp.asarray([[1., 1.], [.5, .5], [.5, .5], [1e-6, 1e-6]], dtype=jnp.float64)
        state = TransportState(density=density, pressure=12.5 * density, Er=jnp.zeros(2))
        system = ComposedEquationSystem(
            equations=(), density_floor=1e-6, temperature_floor=1e-6,
            temperature_active_mask=jnp.asarray([True, True, True, False]),
            fixed_temperature_profile=jnp.full((4, 2), 12.5, dtype=jnp.float64),
        )
        pressure_bar = jnp.full((4, 2), 3e307, dtype=jnp.float64).at[3].set(0.)
        incoming = TransportState(density=jnp.zeros_like(density), pressure=pressure_bar, Er=jnp.zeros(2))
        # This is the same preparation transpose reached by the database
        # direct-state RHS pullback, rather than a separately copied formula.
        operation = jax.jit(system._prepare_working_state_pullback)
        actual = jax.device_get(operation(state, incoming))
        np.testing.assert_array_equal(actual.density, np.zeros_like(density))
        np.testing.assert_array_equal(actual.pressure, np.asarray(pressure_bar))
        np.testing.assert_array_equal(actual.Er, np.zeros(2))

    def test_disabled_temperature_floor_is_a_noop(self):
        state = TransportState(density=jnp.zeros((1, 2)), pressure=jnp.ones((1, 2)), Er=jnp.zeros(2))
        self.assertIs(apply_transport_temperature_floor(state, temperature_floor=None), state)


if __name__ == "__main__":
    unittest.main()
