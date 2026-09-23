import dataclasses
from types import SimpleNamespace

import jax
import jax.numpy as jnp

from NEOPAX._state import (
    TransportState,
    apply_transport_density_floor,
    apply_transport_temperature_floor,
    safe_density,
    safe_temperature,
)
from NEOPAX._transport_flux_models import (
    PRESSURE_SOURCE_STATE_TO_MW_M3,
    compute_net_total_power_volume_average_mw_m3,
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


def test_net_total_power_volume_average_is_signed_and_jittable():
    state = TransportState(
        density=jnp.ones((1, 3)),
        pressure=jnp.asarray([[2.0, 4.0, 8.0]]),
        Er=jnp.zeros(3),
    )
    geometry = SimpleNamespace(
        Vprime=jnp.asarray([1.0, 2.0, 1.0]),
        r_grid=jnp.asarray([0.0, 0.5, 1.0]),
    )

    def pressure_sources(value):
        return {
            "AlphaPower": value.pressure[0],
            "PBrems": 0.5 * value.pressure[0],
            "external_heating": jnp.ones(3),
        }

    value = compute_net_total_power_volume_average_mw_m3(
        state, pressure_sources, geometry
    )
    net_source = 0.5 * state.pressure[0] + 1.0
    expected = PRESSURE_SOURCE_STATE_TO_MW_M3 * (
        jnp.trapezoid(net_source * geometry.Vprime, x=geometry.r_grid)
        / jnp.trapezoid(geometry.Vprime, x=geometry.r_grid)
    )
    assert jnp.allclose(value, expected)

    jitted = jax.jit(
        lambda pressure: compute_net_total_power_volume_average_mw_m3(
            dataclasses.replace(state, pressure=pressure), pressure_sources, geometry
        )
    )
    assert jnp.allclose(jitted(state.pressure), expected)
