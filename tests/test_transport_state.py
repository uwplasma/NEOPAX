import jax.numpy as jnp

from NEOPAX._state import (
    TransportState,
    apply_transport_density_floor,
    apply_transport_temperature_floor,
    safe_density,
    safe_temperature,
)


def test_transport_state_temperature_property_uses_safe_density():
    state = TransportState(
        density=jnp.array([[0.0, 2.0]]),
        pressure=jnp.array([[4.0, 6.0]]),
        Er=jnp.zeros(2),
    )
    temperature = state.temperature
    assert temperature.shape == (1, 2)
    assert jnp.isfinite(temperature).all()


def test_safe_density_broadcasts_species_floor():
    density = jnp.array([[0.01, 0.20], [0.30, 0.01]])
    out = safe_density(density, jnp.array([0.1, 0.05]))
    expected = jnp.array([[0.1, 0.20], [0.30, 0.05]])
    assert jnp.allclose(out, expected)


def test_safe_temperature_none_floor_is_noop():
    temperature = jnp.array([[1.0, 2.0]])
    assert jnp.allclose(safe_temperature(temperature, None), temperature)


def test_apply_transport_density_floor_updates_state():
    state = TransportState(
        density=jnp.array([[0.01, 0.20]]),
        pressure=jnp.array([[1.0, 2.0]]),
        Er=jnp.zeros(2),
    )
    floored = apply_transport_density_floor(state, 0.1)
    assert jnp.allclose(floored.density, jnp.array([[0.1, 0.20]]))


def test_apply_transport_temperature_floor_updates_pressure_consistently():
    state = TransportState(
        density=jnp.array([[1.0, 2.0]]),
        pressure=jnp.array([[0.2, 10.0]]),
        Er=jnp.zeros(2),
    )
    floored = apply_transport_temperature_floor(state, temperature_floor=1.0, density_floor=1.0e-6)
    assert jnp.allclose(floored.temperature, jnp.array([[1.0, 5.0]]))
    assert jnp.allclose(floored.pressure, jnp.array([[1.0, 10.0]]))
