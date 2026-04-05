import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from NEOPAX._boundary_conditions import BoundaryConditionModel, right_constraints_from_bc_model
from NEOPAX._cell_variable import make_profile_cell_variable
from NEOPAX._fem import conservative_update
from NEOPAX._parameters import Solver_Parameters
from NEOPAX._transport_solvers import ThetaNewtonSolver, RADAUSolver, build_time_solver


@dataclasses.dataclass
class MockField:
    n_r: int
    r_grid_half: jax.Array
    r_grid: jax.Array
    Vprime: jax.Array
    Vprime_half: jax.Array


def make_mock_field(n_r: int) -> MockField:
    r_half = jnp.linspace(0.0, 1.0, n_r + 1)
    r_cell = 0.5 * (r_half[1:] + r_half[:-1])
    return MockField(
        n_r=n_r,
        r_grid_half=r_half,
        r_grid=r_cell,
        Vprime=jnp.ones(n_r),
        Vprime_half=jnp.ones(n_r + 1),
    )


def test_conservative_update_scalar_vs_per_cell_dx_uniform_grid():
    field = make_mock_field(16)
    flux = jnp.linspace(0.0, 1.0, field.n_r + 1)
    dx_scalar = 1.0 / field.n_r
    dx_cells = jnp.diff(field.r_grid_half)

    rhs_scalar = conservative_update(flux, dx_scalar, field.Vprime, field.Vprime_half)
    rhs_cells = conservative_update(flux, dx_cells, field.Vprime, field.Vprime_half)

    np.testing.assert_allclose(rhs_cells, rhs_scalar, rtol=1e-12, atol=1e-12)


def test_right_constraints_from_bc_model_all_types():
    default_val = jnp.array([2.0, 3.0])

    bc_dirichlet = BoundaryConditionModel(dr=1.0, right_type="dirichlet", right_value=jnp.array([5.0, 7.0]))
    rv, rg = right_constraints_from_bc_model(bc_dirichlet, default_val)
    assert rg is None
    np.testing.assert_allclose(rv, jnp.array([5.0, 7.0]))

    bc_neumann = BoundaryConditionModel(dr=1.0, right_type="neumann", right_gradient=jnp.array([0.4, -0.2]))
    rv, rg = right_constraints_from_bc_model(bc_neumann, default_val)
    assert rv is None
    np.testing.assert_allclose(rg, jnp.array([0.4, -0.2]))

    bc_robin = BoundaryConditionModel(
        dr=1.0,
        right_type="robin",
        right_value=jnp.array([2.0, 3.0]),
        right_decay_length=jnp.array([4.0, 6.0]),
    )
    rv, rg = right_constraints_from_bc_model(bc_robin, default_val)
    assert rv is None
    np.testing.assert_allclose(rg, jnp.array([-0.5, -0.5]), rtol=1e-12, atol=1e-12)


def test_zero_divergence_for_constant_flux_flat_geometry():
    field = make_mock_field(32)
    const_flux = jnp.full(field.n_r, 3.7)
    cv = make_profile_cell_variable(
        const_flux,
        field.r_grid_half,
        left_face_constraint=jnp.asarray(3.7),
        right_face_constraint=jnp.asarray(3.7),
    )
    face_flux = cv.face_value()
    rhs = conservative_update(face_flux, jnp.diff(field.r_grid_half), field.Vprime, field.Vprime_half)

    np.testing.assert_allclose(rhs, jnp.zeros_like(rhs), atol=1e-12)


def test_fvm_rhs_jittable_and_differentiable():
    field = make_mock_field(16)

    def rhs(u):
        cv = make_profile_cell_variable(
            u,
            field.r_grid_half,
            left_face_constraint=jnp.asarray(0.0, dtype=u.dtype),
            right_face_constraint=jnp.asarray(0.0, dtype=u.dtype),
        )
        flux_faces = cv.face_grad()
        return conservative_update(flux_faces, jnp.diff(field.r_grid_half), field.Vprime, field.Vprime_half)

    u0 = jnp.sin(jnp.pi * field.r_grid)

    rhs_jit = jax.jit(rhs)
    out = rhs_jit(u0)
    assert out.shape == (field.n_r,)

    grad_fn = jax.jit(jax.grad(lambda u: jnp.sum(rhs(u))))
    g = grad_fn(u0)
    assert g.shape == (field.n_r,)
    assert jnp.all(jnp.isfinite(g))


def _diffusion_rhs(y, field, D):
    """1D diffusion RHS in conservative form: du/dt = -div(-D grad u)."""
    cv = make_profile_cell_variable(
        y,
        field.r_grid_half,
        left_face_constraint=jnp.asarray(0.0, dtype=y.dtype),
        right_face_constraint=jnp.asarray(0.0, dtype=y.dtype),
        left_face_grad_constraint=None,
        right_face_grad_constraint=None,
    )
    flux_faces = -D * cv.face_grad()
    return conservative_update(flux_faces, jnp.diff(field.r_grid_half), field.Vprime, field.Vprime_half)


def test_diffusion_spatial_convergence_manufactured_solution():
    """Check spatial convergence order for the diffusion operator on sin(pi r)."""
    D = 1.0
    n_r_list = [16, 32, 64, 128]
    errors = []

    for n_r in n_r_list:
        field = make_mock_field(n_r)
        r = field.r_grid
        u = jnp.sin(jnp.pi * r)
        rhs_num = _diffusion_rhs(u, field, D)
        rhs_exact = -D * jnp.pi**2 * jnp.sin(jnp.pi * r)
        errors.append(float(jnp.sqrt(jnp.mean((rhs_num - rhs_exact) ** 2))))

    rates = [np.log(errors[i] / errors[i + 1]) / np.log(2.0) for i in range(len(errors) - 1)]
    assert all(rate > 1.5 for rate in rates), f"Expected near 2nd order, got rates={rates}"


def _solve_diffusion_with_backend(backend, y0, t0, tf, dt, field, D, *, rtol=1e-6, atol=1e-8):
    """Solve diffusion toy PDE with one of the existing time-solver backends."""
    vector_field = lambda t, y: _diffusion_rhs(y, field, D)

    if backend == "theta":
        solver = ThetaNewtonSolver(
            t0=t0,
            t1=tf,
            dt=dt,
            theta_implicit=1.0,
            tol=1.0e-9,
            maxiter=20,
            ptc_enabled=True,
            line_search_enabled=True,
            max_step_retries=8,
            differentiable_mode=False,
        )
        out = solver.solve(y0, vector_field)
        return out["final_state"]

    if backend == "radau":
        solver = RADAUSolver(t0=t0, t1=tf, dt=dt, rtol=rtol, atol=atol)
        out = solver.solve(y0, vector_field)
        return out["final_state"]

    if backend == "kvaerno5":
        pytest.importorskip("diffrax")
        ts = jnp.linspace(t0, tf, 128)
        params = Solver_Parameters(
            t0=t0,
            t_final=tf,
            dt=dt,
            ts_list=ts,
            rtol=rtol,
            atol=atol,
            integrator="diffrax_kvaerno5",
            transport_solver_family="ode",
            transport_solver_backend="diffrax_kvaerno5",
        )
        solver = build_time_solver(params)
        sol = solver.solve(y0, vector_field)
        return sol.ys[-1]

    raise ValueError(f"Unsupported backend: {backend}")


@pytest.mark.parametrize("backend,dt,tol", [
    # Theta (fully implicit, current settings) is less accurate here than adaptive RK-implicit backends.
    ("theta", 2.5e-3, 1.5e-2),
    ("radau", 1.0e-2, 2e-4),
    ("kvaerno5", 1.0e-2, 2e-4),
])
def test_diffusion_end_to_end_with_existing_solvers(backend, dt, tol):
    """End-to-end diffusion toy model using already-tested time solvers."""
    n_r = 64
    t0, tf = 0.0, 0.05
    D = 1.0
    field = make_mock_field(n_r)
    y0 = jnp.sin(jnp.pi * field.r_grid)
    y_exact = jnp.exp(-D * jnp.pi**2 * tf) * jnp.sin(jnp.pi * field.r_grid)

    y_final = _solve_diffusion_with_backend(backend, y0, t0, tf, dt, field, D)
    l2_error = float(jnp.sqrt(jnp.mean((y_final - y_exact) ** 2)))
    assert l2_error < tol, f"backend={backend}, L2={l2_error:.3e}, tol={tol:.1e}"


def test_weno3_reduces_overshoot_on_steep_profile():
    """WENO3 reconstruction should not overshoot more than linear on steep fronts."""
    field = make_mock_field(128)
    r = field.r_grid
    # Sharp but smooth profile in [0, 1]
    profile = 0.5 * (1.0 + jnp.tanh(80.0 * (r - 0.5)))

    cv = make_profile_cell_variable(
        profile,
        field.r_grid_half,
        left_face_constraint=jnp.asarray(0.0),
        right_face_constraint=jnp.asarray(1.0),
    )

    fv_linear = cv.face_value(reconstruction="linear")
    fv_weno = cv.face_value(reconstruction="weno3")

    overshoot_linear = jnp.sum(jnp.maximum(fv_linear - 1.0, 0.0) + jnp.maximum(-fv_linear, 0.0))
    overshoot_weno = jnp.sum(jnp.maximum(fv_weno - 1.0, 0.0) + jnp.maximum(-fv_weno, 0.0))

    assert float(overshoot_weno) <= float(overshoot_linear) + 1.0e-14


def test_weno3_matches_linear_on_smooth_profile():
    """On smooth data, WENO3 should stay close to linear reconstruction."""
    field = make_mock_field(256)
    r = field.r_grid
    profile = 0.2 + 0.8 * jnp.sin(jnp.pi * r)

    cv = make_profile_cell_variable(
        profile,
        field.r_grid_half,
        left_face_constraint=jnp.asarray(0.2),
        right_face_constraint=jnp.asarray(0.2),
    )

    fv_linear = cv.face_value(reconstruction="linear")
    fv_weno = cv.face_value(reconstruction="weno3")
    l2 = jnp.sqrt(jnp.mean((fv_weno - fv_linear) ** 2))

    assert float(l2) < 2.0e-3
