"""
Broyden's method for quasi-Newton root-finding, inspired by torax.
"""
import jax
import jax.numpy as jnp
from typing import Callable, Any


class BroydenSolver:
    def __init__(self, tol=1e-8, maxiter=100):
        self.tol = tol
        self.maxiter = maxiter

    def solve(self, x0, func: Callable, *args, **kwargs) -> Any:
        tol, maxiter = self.tol, self.maxiter
        flat_x0, unravel = jax.flatten_util.ravel_pytree(x0)
        n = flat_x0.shape[0]
        B0 = jnp.eye(n)
        fx0 = func(x0, *args, **kwargs)
        flat_fx0, _ = jax.flatten_util.ravel_pytree(fx0)

        def cond_fun(state):
            k, flat_x, B, flat_fx = state
            return (k < maxiter) & (jnp.linalg.norm(flat_fx) > tol)

        def body_fun(state):
            k, flat_x, B, flat_fx = state
            dx = -jax.scipy.linalg.solve(B, flat_fx)
            flat_x_new = flat_x + dx
            x_new = unravel(flat_x_new)
            fx_new = func(x_new, *args, **kwargs)
            flat_fx_new, _ = jax.flatten_util.ravel_pytree(fx_new)
            y = flat_fx_new - flat_fx
            s = dx
            B_new = B + jnp.outer((y - B @ s), s) / (s @ s)
            return (k + 1, flat_x_new, B_new, flat_fx_new)

        init_state = (0, flat_x0, B0, flat_fx0)
        k_final, flat_x_final, _, _ = jax.lax.while_loop(cond_fun, body_fun, init_state)
        return unravel(flat_x_final)
