import jax.numpy as jnp
import pytest

from NEOPAX._transport_solvers import (
    DiffraxSolver,
    NewtonThetaMethodSolver,
    RADAUSolver,
    build_time_solver,
)


def _base_solver_parameters(**overrides):
    params = {
        "t0": 0.0,
        "t_final": 1.0,
        "dt": 0.1,
        "transport_solver_backend": "theta_newton",
        "min_step": 1.0e-12,
        "max_step": 0.25,
        "max_steps": 100,
        "theta_implicit": 1.0,
        "nonlinear_solver_tol": 1.0e-8,
        "nonlinear_solver_maxiter": 12,
        "save_n": 4,
        "rtol": 1.0e-6,
        "atol": 1.0e-8,
    }
    params.update(overrides)
    return params


def test_build_time_solver_theta_newton_backend():
    pytest.importorskip("diffrax")
    solver = build_time_solver(_base_solver_parameters(transport_solver_backend="theta_newton"))
    assert isinstance(solver, NewtonThetaMethodSolver)
    assert float(solver.t0) == 0.0
    assert float(solver.t1) == 1.0


def test_build_time_solver_radau_backend():
    pytest.importorskip("diffrax")
    solver = build_time_solver(_base_solver_parameters(transport_solver_backend="radau"))
    assert isinstance(solver, RADAUSolver)
    assert float(solver.t0) == 0.0
    assert float(solver.t1) == 1.0


def test_build_time_solver_legacy_integrator_fallback():
    pytest.importorskip("diffrax")
    solver = build_time_solver(_base_solver_parameters(transport_solver_backend="diffrax_kvaerno5"))
    assert isinstance(solver, DiffraxSolver)
    assert float(solver.t0) == 0.0
    assert float(solver.t1) == 1.0


def test_build_time_solver_accepts_solver_instance_override():
    pytest.importorskip("diffrax")
    override = RADAUSolver(t0=0.0, t1=2.0, dt=0.2)
    solver = build_time_solver(_base_solver_parameters(), solver_override=override)
    assert solver is override


def test_theta_newton_solver_runs_scalar_decay_problem():
    solver = NewtonThetaMethodSolver(
        t0=0.0,
        t1=0.5,
        dt=0.05,
        theta_implicit=1.0,
        tol=1.0e-9,
        maxiter=20,
        max_steps=128,
        save_n=4,
    )

    def vector_field(t, y):
        del t
        return -2.0 * y

    out = solver.solve(jnp.array([1.0]), vector_field)
    final_state = out["final_state"]
    assert jnp.all(jnp.isfinite(final_state))
    assert final_state.shape == (1,)
    assert float(final_state[0]) < 1.0
