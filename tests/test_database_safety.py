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


def test_3d_array_loader_preserves_scaled_values_and_raw_data_ad():
    import jax
    import numpy as np
    from NEOPAX._database import Monoenergetic3D
    from NEOPAX._interpolators import get_Dij_3d

    rho = jnp.linspace(.1, .9, 5)
    nu = jnp.asarray([1e-5, 1e-3, .1])
    er = jnp.broadcast_to(jnp.asarray([0., 1e-5, 1e-3, .1]), (5, 4))
    drds = jnp.linspace(.2, .4, 5)
    d11 = jnp.linspace(1e-4, 2e-4, 60).reshape((5, 3, 4))
    d13, d33 = d11 * 100., d11 * 200.
    original = np.asarray(d11).copy()

    def build(step):
        return Monoenergetic3D.read_data(2. + step, rho, nu, er, drds * (1. + step),
                                         d11, d13 * (1. + step), d33)

    database = build(jnp.asarray(0.))
    np.testing.assert_allclose(10.**database.D11_log, np.asarray(d11 * drds[:, None, None]**2))
    np.testing.assert_allclose(database.D13, np.asarray(d13 * drds[:, None, None]))
    np.testing.assert_allclose(database.D33, np.asarray(d33 * nu[None, :, None]))
    np.testing.assert_allclose(database.Er_list, np.log10(np.maximum(
        1e-8, np.abs(np.asarray(er)[0])[None, :] / (2. * np.asarray(rho)[:, None]))))
    np.testing.assert_array_equal(d11, original)

    objective = lambda step: jnp.sum(get_Dij_3d(.73, .002, .0004, build(step)))
    tangent = jax.jit(jax.grad(objective))(jnp.asarray(0.))
    fd = (objective(1e-5) - objective(-1e-5)) / 2e-5
    assert np.isfinite(tangent)
    np.testing.assert_allclose(tangent, fd, rtol=3e-7, atol=3e-8)
