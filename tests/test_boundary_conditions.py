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


def test_robin_honours_boundary_value_and_recovers_limits():
    """Inhomogeneous Robin: u + lambda du/dn = value.

    ``value`` used to be accepted by the config schema (NEOPAX's own examples set
    it) but ignored by every robin code path, so a caller imposing a boundary
    value silently got the homogeneous decay condition instead.
    """
    arr = jnp.array([[3.0, 2.0, 1.0]])
    dr = 0.5
    target = 0.2

    # lambda -> 0 reproduces Dirichlet(value)
    bc_tight = BoundaryConditionModel(
        dr=dr, left_type="neumann", right_type="robin",
        right_value=jnp.array([target]), right_decay_length=jnp.array([1e-9]),
    )
    edge_tight = bc_tight.apply_ghost_all(arr)[0, -1]
    assert jnp.allclose(edge_tight, target, atol=1e-6)

    # lambda -> inf reproduces zero-gradient (Neumann): edge equals its neighbour
    bc_loose = BoundaryConditionModel(
        dr=dr, left_type="neumann", right_type="robin",
        right_value=jnp.array([target]), right_decay_length=jnp.array([1e9]),
    )
    edge_loose = bc_loose.apply_ghost_all(arr)[0, -1]
    assert jnp.allclose(edge_loose, arr[0, -1], atol=1e-6)

    # a finite lambda lands strictly between the two limits
    bc_mid = BoundaryConditionModel(
        dr=dr, left_type="neumann", right_type="robin",
        right_value=jnp.array([target]), right_decay_length=jnp.array([dr]),
    )
    edge_mid = bc_mid.apply_ghost_all(arr)[0, -1]
    assert bool(jnp.all(edge_mid > target)) and bool(jnp.all(edge_mid < arr[0, -1]))


def test_robin_without_value_is_unchanged():
    """Omitting ``value`` must reproduce the previous homogeneous decay exactly."""
    arr = jnp.array([[3.0, 2.0, 1.0]])
    dr, decay = 0.5, 0.25
    bc = BoundaryConditionModel(
        dr=dr, left_type="neumann", right_type="robin",
        right_decay_length=jnp.array([decay]),
    )
    expected = arr[0, -1] + (-arr[0, -1] / decay) * dr    # old formula, ref == arr
    assert jnp.allclose(bc.apply_ghost_all(arr)[0, -1], expected, atol=1e-8)


def test_right_constraints_robin_value_reduces_to_homogeneous():
    """The FVM face path gains the same generalization, and is exactly backward
    compatible at ``value = 0``."""
    default = jnp.array([1.0])
    profile = jnp.array([[3.0, 2.0, 1.0]])
    faces = jnp.array([0.0, 0.5, 1.0, 1.5])
    decay = jnp.array([0.4])

    bc_hom = BoundaryConditionModel(dr=0.5, right_type="robin", right_decay_length=decay)
    rv_hom, _ = right_constraints_from_bc_model(bc_hom, default, profile=profile,
                                                face_centers=faces)
    bc_zero = BoundaryConditionModel(dr=0.5, right_type="robin",
                                     right_value=jnp.array([0.0]),
                                     right_decay_length=decay)
    rv_zero, _ = right_constraints_from_bc_model(bc_zero, default, profile=profile,
                                                 face_centers=faces)
    assert jnp.allclose(rv_hom, rv_zero, atol=1e-10)

    # a positive target raises the face value above the homogeneous one
    bc_val = BoundaryConditionModel(dr=0.5, right_type="robin",
                                    right_value=jnp.array([5.0]),
                                    right_decay_length=decay)
    rv_val, rg_val = right_constraints_from_bc_model(bc_val, default, profile=profile,
                                                     face_centers=faces)
    assert bool(jnp.all(rv_val > rv_hom))
    assert jnp.allclose(rg_val, (5.0 - rv_val) / decay, atol=1e-8)
