# JAX-compatible robust Newton-Raphson root-finder (torax-style)
# Adapted for NEOPAX ambipolarity root-finding
import jax
import jax.numpy as jnp
from jax import lax
from typing import Callable

def root_newton_raphson(
    fun: Callable[[jnp.ndarray], jnp.ndarray],
    x0: jnp.ndarray,
    maxiter: int = 30,
    tol: float = 1e-5,
    coarse_tol: float = 1e-2,
    delta_reduction_factor: float = 0.5,
    tau_min: float = 0.01,
    log_iterations: bool = False,
    custom_jac: Callable[[jnp.ndarray], jnp.ndarray] = None,
):
    """
    Differentiable Newton-Raphson root finder (vector/scalar), with line search and robust fallback.
    Returns (x_root, metadata_dict)
    """
    def norm(x):
        return jnp.mean(jnp.abs(x))

    def jacobian(f, x):
        return custom_jac(x) if custom_jac is not None else jax.grad(f) if x.ndim == 0 else jax.jacfwd(f)(x)

    def body(state):
        x, fx, tau, iters = state
        J = jacobian(fun, x)
        # Newton step
        if x.ndim == 0:
            delta = fx / J
        else:
            delta = jnp.linalg.solve(J, fx)
        x_new = x - tau * delta
        fx_new = fun(x_new)
        # Line search: reduce tau if not improved
        tau_new = lax.cond(norm(fx_new) < norm(fx), lambda _: 1.0, lambda _: delta_reduction_factor * tau, None)
        return (x_new, fx_new, tau_new, iters + 1)

    def cond(state):
        x, fx, tau, iters = state
        return (norm(fx) > tol) & (iters < maxiter) & (tau > tau_min)

    fx0 = fun(x0)
    state = (x0, fx0, 1.0, 0)
    state = lax.while_loop(cond, body, state)
    x_final, fx_final, tau_final, n_iter = state
    metadata = {
        'converged': norm(fx_final) <= tol,
        'iterations': n_iter,
        'final_residual': float(norm(fx_final)),
        'final_tau': float(tau_final),
    }
    return x_final, metadata
