import jax.numpy as jnp

from NEOPAX._main_solver import _apply_floor_rhs


def test_apply_floor_rhs_scalar_floor():
    profile = jnp.array([0.05, 0.20, 0.03, 0.50])
    rhs = jnp.array([-1.0, -2.0, 3.0, -4.0])

    out = _apply_floor_rhs(rhs, profile, 0.1)

    # Below floor: clamp to non-negative updates.
    assert out[0] == 0.0
    assert out[2] == 3.0
    # Above floor: untouched.
    assert out[1] == -2.0
    assert out[3] == -4.0


def test_apply_floor_rhs_specieswise_floor():
    profile = jnp.array(
        [
            [0.05, 0.07, 0.20],
            [0.60, 0.01, 0.02],
        ]
    )
    rhs = jnp.array(
        [
            [-1.0, 2.0, -3.0],
            [-4.0, -5.0, 6.0],
        ]
    )

    out = _apply_floor_rhs(rhs, profile, jnp.array([0.1, 0.05]))

    expected = jnp.array(
        [
            [0.0, 2.0, -3.0],
            [-4.0, 0.0, 6.0],
        ]
    )
    assert jnp.allclose(out, expected)


def test_apply_floor_rhs_none_is_noop():
    profile = jnp.array([[0.1, 0.2]])
    rhs = jnp.array([[-1.0, 2.0]])
    out = _apply_floor_rhs(rhs, profile, None)
    assert jnp.allclose(out, rhs)
