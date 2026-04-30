import jax.numpy as jnp

from NEOPAX._database import D11_POSITIVE_FLOOR as MONO_D11_FLOOR, _floor_positive_d11
from NEOPAX._database_preprocessed import (
    D11_POSITIVE_FLOOR as PRE_D11_FLOOR,
    _prepare_ntx_arrays,
)


def test_prepare_ntx_arrays_floors_negative_d11_to_small_positive_value():
    rho = jnp.array([0.2, 0.5, 0.8])
    nu_v = jnp.array([1.0e-3, 1.0e-2])
    Er = jnp.array([[0.0, 1.0e-4]])
    drds = jnp.ones_like(rho)
    D11 = jnp.array(
        [
            [[-1.0, 2.0], [0.0, 3.0]],
            [[4.0, 5.0], [6.0, 7.0]],
            [[8.0, 9.0], [10.0, 11.0]],
        ]
    )
    D13 = jnp.zeros_like(D11)
    D33 = jnp.ones_like(D11)

    out = _prepare_ntx_arrays(1.0, rho, nu_v, Er, drds, D11, D13, D33)
    d11 = 10.0 ** out["D11_log"]

    assert jnp.isfinite(out["D11_log"]).all()
    assert jnp.allclose(d11[0, 0, 0], PRE_D11_FLOOR)
    assert jnp.allclose(d11[0, 1, 0], PRE_D11_FLOOR)


def test_database_floor_helper_clips_negative_d11_to_small_positive_value():
    d11 = jnp.array([[-3.0, 0.0, 2.0]])
    out = _floor_positive_d11(d11)

    assert jnp.all(out > 0.0)
    assert jnp.allclose(out[0, 0], MONO_D11_FLOOR)
    assert jnp.allclose(out[0, 1], MONO_D11_FLOOR)
    assert jnp.allclose(out[0, 2], 2.0)
