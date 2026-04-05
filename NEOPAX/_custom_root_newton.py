import jax
import jax.numpy as jnp
from jax import lax

# Fully JAX-native, torax-style root finder using jax.lax.custom_root
# This is a scalar version for ambipolarity, but can be extended to vector roots

def newton_raphson_custom_root(fun, x0, tol=1e-6, maxiter=30, use_custom_root=False):
    """
    JAX-native Newton-Raphson root finder with implicit differentiation support.
    Uses jax.lax.custom_root for full JAX compatibility.
    """
    def solve(f, x0):
        def body(state):
            x, fx, i = state
            dfx = jax.grad(f)(x)
            x_new = x - fx / dfx
            fx_new = f(x_new)
            return (x_new, fx_new, i + 1)
        def cond(state):
            x, fx, i = state
            return (jnp.abs(fx) > tol) & (i < maxiter)
        fx0 = f(x0)
        state = (x0, fx0, 0)
        x_final, fx_final, n_iter = lax.while_loop(cond, body, state)
        return x_final

    def tangent_solve(g, y):
        # JAX passes the linearized function g and RHS y.
        # For scalar roots, solve g(x) = y via x = y / g(1).
        return y / g(jnp.ones_like(y))

    # Torax-inspired tradeoff: disabling custom_root can significantly reduce
    # compile time when implicit differentiation through the root solve is not needed.
    if use_custom_root:
        return lax.custom_root(fun, x0, solve, tangent_solve)
    return solve(fun, x0)
