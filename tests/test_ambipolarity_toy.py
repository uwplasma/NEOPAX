"""
Test for NEOPAX._ambipolarity root-finder using a toy model similar to stellarator Er ambipolarity.
"""
import jax
import jax.numpy as jnp
import numpy as np
from NEOPAX._ambipolarity import find_all_ambipolar_Er_roots_min_entropy_jit

def toy_gamma_func(er):
    # Toy model: sum_s q_s Gamma_s(Er) = tanh(2*Er) - 0.5*Er
    return jnp.tanh(2 * er) - 0.5 * er

def toy_entropy_func(er):
    # Toy entropy: |Gamma(Er)|
    return jnp.abs(toy_gamma_func(er))

def test_toy_ambipolar_roots():
    # True roots: solve tanh(2*Er) - 0.5*Er = 0
    # There is always a root at Er=0, and possibly others for larger |Er|
    roots, entropies, best_root, n_roots = find_all_ambipolar_Er_roots_min_entropy_jit(
        toy_gamma_func,
        toy_entropy_func,
        Er_range=(-4.0, 4.0),
        n_scan=32,
        tol=1e-8,
        x_tol=1e-8,
        maxiter=15,
        dtype=jnp.float64,
    )
    roots_np = np.array(roots[:n_roots])
    # Check that a root near zero exists
    assert np.any(np.abs(roots_np) < 1e-6), f"No root near zero found: {roots_np}"
    # Check that all roots satisfy the equation to tolerance
    assert np.all(np.abs(np.tanh(2*roots_np) - 0.5*roots_np) < 1e-7), f"Roots do not satisfy equation: {roots_np}"
    # Check that the minimum-entropy root is the one near zero
    assert np.abs(best_root) < 1.0, f"Best root not near zero: {best_root}"

def test_toy_ambipolar_profile():
    # Test a radial profile of toy roots
    n_r = 8
    coeffs = np.linspace(0.3, 1.0, n_r)
    roots_profile = np.zeros(n_r)
    for i, c in enumerate(coeffs):
        def gamma_func(er):
            return jnp.tanh(2 * er) - c * er
        def entropy_func(er):
            return jnp.abs(gamma_func(er))
        roots, entropies, best_root, n_roots = find_all_ambipolar_Er_roots_min_entropy_jit(
            gamma_func,
            entropy_func,
            Er_range=(-4.0, 4.0),
            n_scan=32,
            tol=1e-8,
            x_tol=1e-8,
            maxiter=15,
            dtype=jnp.float64,
        )
        roots_np = np.array(roots[:n_roots])
        # Should always be a root near zero
        assert np.any(np.abs(roots_np) < 1e-6), f"No root near zero for c={c}: {roots_np}"
        roots_profile[i] = best_root
    # Profile should be smooth and monotonic
    assert np.all(np.diff(roots_profile) >= -1e-6), f"Profile not monotonic: {roots_profile}"
