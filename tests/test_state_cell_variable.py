import jax.numpy as jnp
import numpy as np

from NEOPAX._cell_variable import CellVariable, get_gradient_temperature, make_profile_cell_variable


def test_face_value_uses_boundary_constraints():
    face_centers = jnp.array([0.0, 0.5, 1.0, 1.5])
    var = make_profile_cell_variable(
        jnp.array([1.0, 3.0, 5.0]),
        face_centers,
        left_face_grad_constraint=0.0,
        right_face_constraint=7.0,
    )

    np.testing.assert_allclose(var.face_value(), jnp.array([1.0, 2.0, 4.0, 7.0]))


def test_grad_matches_linear_profile_on_uniform_grid():
    face_centers = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    cell_centers = 0.5 * (face_centers[1:] + face_centers[:-1])
    values = 2.0 * cell_centers + 1.0
    right_face_value = 2.0 * face_centers[-1] + 1.0

    var = CellVariable(
        value=values,
        face_centers=face_centers,
        left_face_grad_constraint=jnp.array(2.0),
        right_face_constraint=jnp.array(right_face_value),
    )

    np.testing.assert_allclose(var.grad(), jnp.full_like(values, 2.0), rtol=1e-7)


def test_get_gradient_temperature_respects_edge_value():
    r_grid = jnp.array([0.5, 1.5, 2.5])
    r_grid_half = jnp.array([0.0, 1.0, 2.0, 3.0])
    temperature = jnp.array([10.0, 8.0, 6.0])

    grad = get_gradient_temperature(temperature, r_grid, r_grid_half, 1.0, 4.0)

    # With linear face interpolation plus right Dirichlet face constraint (4.0),
    # the face values are [10, 9, 7, 4] and per-cell gradients are [-1, -2, -3].
    np.testing.assert_allclose(grad, jnp.array([-1.0, -2.0, -3.0]), rtol=1e-7)


def test_face_value_weno3_shape_and_boundary_constraints():
    face_centers = jnp.array([0.0, 0.5, 1.0, 1.5, 2.0])
    var = make_profile_cell_variable(
        jnp.array([0.0, 1.0, 0.2, 1.2]),
        face_centers,
        left_face_constraint=0.0,
        right_face_constraint=1.0,
    )

    fv_linear = var.face_value(reconstruction="linear")
    fv_weno = var.face_value(reconstruction="weno3")

    assert fv_weno.shape == fv_linear.shape
    assert jnp.isclose(fv_weno[0], 0.0)
    assert jnp.isclose(fv_weno[-1], 1.0)


def test_face_value_weno3_is_bounded_by_neighbor_cells():
    face_centers = jnp.linspace(0.0, 1.0, 7)
    values = jnp.array([0.0, 2.0, -1.0, 3.0, 0.5, 1.5])
    var = make_profile_cell_variable(
        values,
        face_centers,
        left_face_constraint=values[0],
        right_face_constraint=values[-1],
    )

    fv_weno = var.face_value(reconstruction="weno3")
    inner = fv_weno[1:-1]
    lower = jnp.minimum(values[:-1], values[1:])
    upper = jnp.maximum(values[:-1], values[1:])

    assert jnp.all(inner >= lower - 1e-12)
    assert jnp.all(inner <= upper + 1e-12)