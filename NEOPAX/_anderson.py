"""
Anderson acceleration solver for fixed-point iteration, inspired by torax.
"""
import jax
import jax.numpy as jnp
from typing import Callable, Any


class AndersonSolver:
    def __init__(self, m=5, tol=1e-8, maxiter=100):
        self.m = m
        self.tol = tol
        self.maxiter = maxiter

    def solve(self, x0, func: Callable, *args, **kwargs) -> Any:
        m, tol, maxiter = self.m, self.tol, self.maxiter
        x_shape = x0.shape
        X = jnp.zeros((m, *x_shape)).at[0].set(x0)
        F = jnp.zeros((m, *x_shape)).at[0].set(func(x0, *args, **kwargs) - x0)
        k0 = 1
        def cond_fun(state):
            k, x, X, F = state
            f_new = func(x, *args, **kwargs) - x
            return (k < maxiter) & (jnp.linalg.norm(f_new) > tol)

        def body_fun(state):
            k, x, X, F = state
            x_new = func(x, *args, **kwargs)
            f_new = x_new - x
            X = X.at[k % m].set(x_new)
            F = F.at[k % m].set(f_new)
            idxs = jnp.arange(m)
            valid = idxs < jnp.minimum(k + 1, m)
            G = jnp.where(valid[:, None], F, 0)
            dG = G[1:] - G[:-1]
            dX = X[1:] - X[:-1]
            # Use lstsq for gamma, fallback to zeros if ill-conditioned
            gamma = jax.scipy.linalg.lstsq(dG.T, f_new.T, rcond=None)[0]
            dx = -f_new + jnp.dot(gamma, dX)
            x_next = x_new + dx
            return (k + 1, x_next, X, F)

        k_final, x_final, _, _ = jax.lax.while_loop(cond_fun, body_fun, (k0, x0, X, F))
        return x_final
