import jax
import jax.numpy as jnp
import pytest
from NEOPAX._boundary_conditions import (
    DirichletBC,
    NeumannBC,
    BoundaryConditionModel,
    get_bc,
    register_bc,
    right_constraints_from_bc_model,
)


def test_dirichlet_bc_apply():
    arr = jnp.ones(5)
    bc = DirichletBC(axis_value=0.0, edge_value=2.0)
    arr_bc = bc.apply(arr)
    assert arr_bc[0] == 0.0
    assert arr_bc[-1] == 2.0


def test_neumann_bc_apply():
    arr = jnp.arange(5.0)
    bc = NeumannBC(grad_axis=1.0, grad_edge=-1.0, dr=1.0)
    arr_bc = bc.apply(arr)
    assert jnp.isclose(arr_bc[0], arr[1] - 1.0)
    assert jnp.isclose(arr_bc[-1], arr[-2] + -1.0)


def test_user_bc_registration():
    class CustomBC(DirichletBC):
        pass
    register_bc('custom', CustomBC)
    bc = get_bc('custom', axis_value=3.0, edge_value=4.0)
    arr = jnp.ones(5)
    arr_bc = bc.apply(arr)
    assert arr_bc[0] == 3.0
    assert arr_bc[-1] == 4.0


def test_jax_jit_compatibility():
    bc = DirichletBC(axis_value=0.0, edge_value=1.0)
    arr = jnp.ones(5)
    jitted_apply = jax.jit(bc.apply)
    arr_bc = jitted_apply(arr)
    assert arr_bc[0] == 0.0
    assert arr_bc[-1] == 1.0


def test_error_on_wrong_shape():
    bc = DirichletBC(axis_value=0.0, edge_value=1.0)
    arr = jnp.ones((2, 5))
    arr_bc = bc.apply(arr)
    assert arr_bc.shape == (2, 5)


def test_right_constraints_dirichlet_neumann_robin():
    default = jnp.array([2.0, 3.0])

    bc_dirichlet = BoundaryConditionModel(dr=1.0, right_type="dirichlet", right_value=jnp.array([4.0, 5.0]))
    rv, rg = right_constraints_from_bc_model(bc_dirichlet, default)
    assert rg is None
    assert jnp.allclose(rv, jnp.array([4.0, 5.0]))

    bc_neumann = BoundaryConditionModel(dr=1.0, right_type="neumann", right_gradient=jnp.array([0.1, -0.2]))
    rv, rg = right_constraints_from_bc_model(bc_neumann, default)
    assert rv is None
    assert jnp.allclose(rg, jnp.array([0.1, -0.2]))

    bc_robin = BoundaryConditionModel(
        dr=1.0,
        right_type="robin",
        right_value=jnp.array([2.0, 3.0]),
        right_decay_length=jnp.array([4.0, 6.0]),
    )
    rv, rg = right_constraints_from_bc_model(bc_robin, default)
    assert rv is None
    assert jnp.allclose(rg, jnp.array([-0.5, -0.5]))


def test_right_constraints_unsupported_type_raises():
    bc_bad = BoundaryConditionModel(dr=1.0, right_type="unsupported")
    with pytest.raises(ValueError):
        right_constraints_from_bc_model(bc_bad, jnp.array([1.0]))
