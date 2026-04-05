import jax
import jax.numpy as jnp
import pytest
from NEOPAX.NEOPAX._transport_solvers import (
    DiffraxSolver, NewtonSolver, AndersonSolver, BroydenSolver, PredictorCorrectorSolver, JaxoptSteadyStateSolver
)

def test_exponential_decay_diffrax():
    # Non-stiff ODE: dx/dt = -x, x(0) = 1, solution x(t) = exp(-t)
    def vector_field(x, t):
        return -x
    solver = DiffraxSolver(dt=0.01, t0=0.0, t1=1.0)
    result = solver.solve(vector_field, x0=jnp.array(1.0))
    x_final = result['solution'][-1]
    x_exact = jnp.exp(-1.0)
    assert jnp.abs(x_final - x_exact) < 1e-3

def test_stiff_ode_diffrax():
    # Stiff ODE: dx/dt = -1000x + 3000 - 2000*exp(-t), x(0) = 0, solution x(t) = 3 - 0.998*exp(-1000*t) - 2.002*exp(-t)
    def vector_field(x, t):
        return -1000 * x + 3000 - 2000 * jnp.exp(-t)
    solver = DiffraxSolver(dt=0.0001, t0=0.0, t1=0.01)
    result = solver.solve(vector_field, x0=jnp.array(0.0))
    x_final = result['solution'][-1]
    x_exact = 3 - 0.998 * jnp.exp(-1000 * 0.01) - 2.002 * jnp.exp(-0.01)
    assert jnp.abs(x_final - x_exact) < 1e-2

def test_newton_solver():
    # Nonlinear root: x^2 - 2 = 0, root at sqrt(2)
    def f(x):
        return x**2 - 2
    solver = NewtonSolver(tol=1e-6)
    result = solver.solve(f, x0=jnp.array(1.0))
    x_root = result['solution']
    assert jnp.abs(x_root - jnp.sqrt(2)) < 1e-6

def test_anderson_solver():
    # Fixed-point: x = cos(x), solution near 0.739
    def f(x):
        return jnp.cos(x)
    solver = AndersonSolver(tol=1e-6)
    result = solver.solve(f, x0=jnp.array(1.0))
    x_fp = result['solution']
    assert jnp.abs(x_fp - 0.73908513) < 1e-6

def test_broyden_solver():
    # Nonlinear root: x^3 - x - 2 = 0, root near 1.521
    def f(x):
        return x**3 - x - 2
    solver = BroydenSolver(tol=1e-6)
    result = solver.solve(f, x0=jnp.array(1.0))
    x_root = result['solution']
    assert jnp.abs(x_root - 1.5213797) < 1e-6

def test_predictor_corrector_solver():
    # Non-stiff ODE: dx/dt = -x, x(0) = 1, solution x(t) = exp(-t)
    def vector_field(x, t):
        return -x
    solver = PredictorCorrectorSolver(dt=0.01, t0=0.0, t1=1.0)
    result = solver.solve(vector_field, x0=jnp.array(1.0))
    x_final = result['solution'][-1]
    x_exact = jnp.exp(-1.0)
    assert jnp.abs(x_final - x_exact) < 1e-3

def test_jaxopt_solver():
    # Nonlinear root: x^2 - 2 = 0, root at sqrt(2)
    def f(x):
        return x**2 - 2
    solver = JaxoptSteadyStateSolver(tol=1e-6)
    result = solver.solve(f, x0=jnp.array(1.0))
    x_root = result['solution']
    assert jnp.abs(x_root - jnp.sqrt(2)) < 1e-6
