"""Small smoke tests for the self-hosted GPU runner (run with ``pytest -m gpu``)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from NEOPAX._cell_variable import make_profile_cell_variable
from NEOPAX._fem import conservative_update
from tests.test_fvm_discretization import make_mock_field

pytestmark = pytest.mark.gpu


def _rhs(field, u):
    cv = make_profile_cell_variable(
        u,
        field.r_grid_half,
        left_face_constraint=jnp.asarray(0.0, dtype=u.dtype),
        right_face_constraint=jnp.asarray(0.0, dtype=u.dtype),
    )
    return conservative_update(cv.face_grad(), jnp.diff(field.r_grid_half), field.Vprime, field.Vprime_half)


def test_backend_is_gpu():
    assert jax.default_backend() == "gpu"


def test_jitted_rhs_runs_on_gpu_and_matches_cpu():
    field = make_mock_field(64)
    u = jnp.sin(jnp.pi * field.r_grid)
    out = jax.jit(lambda x: _rhs(field, x))(u)
    assert out.devices().pop().platform == "gpu"
    cpu = jax.devices("cpu")[0]
    ref = jax.jit(lambda x: _rhs(field, x), device=cpu)(jax.device_put(u, cpu))
    np.testing.assert_allclose(np.asarray(out), np.asarray(ref), rtol=1e-10, atol=1e-10)


def test_gradient_of_rhs_is_finite_on_gpu():
    field = make_mock_field(64)
    u = jnp.sin(jnp.pi * field.r_grid)
    g = jax.jit(jax.grad(lambda x: jnp.sum(_rhs(field, x) ** 2)))(u)
    assert g.devices().pop().platform == "gpu"
    assert bool(jnp.all(jnp.isfinite(g)))
