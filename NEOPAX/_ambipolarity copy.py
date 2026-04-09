"""
ambipolarity.py: Ambipolarity root-finding and entropy minimization utilities for NEOPAX

This module provides routines to:
- Find all roots of the ambipolarity equation sum_s q_s Gamma_s(Er) = 0
- Evaluate entropy production at each root
- Select the Er root with minimum entropy production (torax-style)

Intended for use in NEOPAX and other transport solvers.
"""


import jax
import jax.numpy as jnp
from jax import lax


def find_ambipolar_Er_min_entropy_jit(
    Gamma_func,
    entropy_func,
    Er_range=(-20.0, 20.0),
    n_scan=96,
    tol=1e-6,
    x_tol=1e-6,
    maxiter=30,
):
    """JIT-friendly ambipolar root search without Python loops.

    Uses a batched safeguarded Newton-bisection update over all scan intervals,
    then selects the minimum-entropy valid root.

    Returns:
      best_root: scalar Er minimizing entropy among valid roots
      roots_all: candidate roots for each interval (shape n_scan-1)
      entropies_all: entropy values with inf for invalid intervals
      valid_mask: boolean mask for valid intervals
    """
    er_min, er_max = float(Er_range[0]), float(Er_range[1])
    er_grid = jnp.linspace(er_min, er_max, n_scan)

    gamma_grid = jax.vmap(Gamma_func)(er_grid)
    left = er_grid[:-1]
    right = er_grid[1:]
    g_left = gamma_grid[:-1]
    g_right = gamma_grid[1:]

    # Candidate intervals: strict sign change or a near-zero endpoint.
    left_zero = jnp.abs(g_left) <= tol
    right_zero = jnp.abs(g_right) <= tol
    sign_change = (g_left * g_right) < 0.0
    valid_init = sign_change | left_zero | right_zero

    x0 = jnp.where(left_zero, left, jnp.where(right_zero, right, 0.5 * (left + right)))

    def body(_, carry):
        x, xl, xr, gl, gr, active = carry
        gx = jax.vmap(Gamma_func)(x)
        dgx = jax.vmap(jax.grad(Gamma_func))(x)

        x_newton = x - gx / (dgx + 1e-12)
        x_bisect = 0.5 * (xl + xr)
        use_newton = active & jnp.isfinite(x_newton) & (x_newton > xl) & (x_newton < xr)
        x_trial = jnp.where(use_newton, x_newton, x_bisect)
        g_trial = jax.vmap(Gamma_func)(x_trial)

        same_sign_as_left = (gl * g_trial) > 0.0
        xl_next = jnp.where(active & same_sign_as_left, x_trial, xl)
        gl_next = jnp.where(active & same_sign_as_left, g_trial, gl)
        xr_next = jnp.where(active & ~same_sign_as_left, x_trial, xr)
        gr_next = jnp.where(active & ~same_sign_as_left, g_trial, gr)

        converged = active & ((jnp.abs(g_trial) <= tol) | (jnp.abs(xr_next - xl_next) <= x_tol))
        active_next = active & ~converged
        return (x_trial, xl_next, xr_next, gl_next, gr_next, active_next)

    carry0 = (x0, left, right, g_left, g_right, valid_init)
    x_final, _, _, _, _, active_final = lax.fori_loop(0, maxiter, body, carry0)

    # Keep initially valid candidates; non-converged values remain as best effort.
    valid_mask = valid_init
    entropy_all = jax.vmap(entropy_func)(x_final)
    entropy_masked = jnp.where(valid_mask, entropy_all, jnp.inf)

    best_idx = jnp.argmin(entropy_masked)
    has_any = jnp.any(valid_mask)
    best_root = jnp.where(has_any, x_final[best_idx], jnp.asarray(0.0, dtype=x_final.dtype))
    return best_root, x_final, entropy_masked, valid_mask


def find_all_ambipolar_Er_roots_min_entropy_jit(
    Gamma_func,
    entropy_func,
    Er_range=(-20.0, 20.0),
    n_scan=24,
    tol=1e-6,
    x_tol=1e-6,
    maxiter=12,
):
    """Two-stage coarse/fine JIT-friendly ambipolar root search.
    Uses a single-stage, moderate grid to bracket all roots, then refines only valid brackets.

    Returns:
      roots: Er roots found (1D array, up to n_scan-1)
      entropies: entropy at each root (1D array)
      best_root: root with minimum entropy
    """
    er_min, er_max = float(Er_range[0]), float(Er_range[1])
    er_grid = jnp.linspace(er_min, er_max, n_scan, dtype=jnp.float64)
    gamma_grid = jax.vmap(Gamma_func)(er_grid)
    left = er_grid[:-1]
    right = er_grid[1:]
    g_left = gamma_grid[:-1]
    g_right = gamma_grid[1:]

    # Bracket intervals with sign change or near-zero endpoint
    left_zero = jnp.abs(g_left) <= tol
    right_zero = jnp.abs(g_right) <= tol
    sign_change = (g_left * g_right) < 0.0
    valid = sign_change | left_zero | right_zero

    # --- DEBUG PRINTS ---
    jax.debug.print("[DEBUG] g_left: {}", g_left)
    jax.debug.print("[DEBUG] g_right: {}", g_right)
    jax.debug.print("[DEBUG] sign_change: {}", sign_change)
    jax.debug.print("[DEBUG] left_zero: {}", left_zero)
    jax.debug.print("[DEBUG] right_zero: {}", right_zero)
    jax.debug.print("[DEBUG] valid: {}", valid)

    max_roots = n_scan - 1

    # Instead of masking, fill invalid brackets with NaN and process all
    def fill_invalid(arr, fill_value=jnp.nan):
        return jnp.where(valid, arr, fill_value)

    xl = fill_invalid(left)
    xr = fill_invalid(right)
    gl = fill_invalid(g_left)
    gr = fill_invalid(g_right)
    n_brackets = jnp.sum(valid)

    def _empty():
        roots = jnp.full((max_roots,), jnp.nan, dtype=jnp.float64)
        entropies = jnp.full((max_roots,), jnp.nan, dtype=jnp.float64)
        best_root = jnp.asarray(0.0, dtype=jnp.float64)
        n_roots = jnp.asarray(0, dtype=jnp.int32)
        return roots, entropies, best_root, n_roots

    def _roots():
        # Initial guess: midpoint
        x = 0.5 * (xl + xr)
        active = valid
        def body(_, carry):
            x, xl, xr, gl, gr, active = carry
            gx = jax.vmap(Gamma_func)(x)
            dgx = jax.vmap(jax.grad(Gamma_func))(x)
            x_newton = x - gx / (dgx + 1e-12)
            x_bisect = 0.5 * (xl + xr)
            use_newton = active & jnp.isfinite(x_newton) & (x_newton > xl) & (x_newton < xr)
            x_trial = jnp.where(use_newton, x_newton, x_bisect)
            g_trial = jax.vmap(Gamma_func)(x_trial)
            same_sign_left = (gl * g_trial) > 0.0
            xl_n = jnp.where(active & same_sign_left,  x_trial, xl)
            gl_n = jnp.where(active & same_sign_left,  g_trial, gl)
            xr_n = jnp.where(active & ~same_sign_left, x_trial, xr)
            gr_n = jnp.where(active & ~same_sign_left, g_trial, gr)
            converged   = active & ((jnp.abs(g_trial) <= tol) | (jnp.abs(xr_n - xl_n) <= x_tol))
            active_next = active & ~converged
            return (x_trial, xl_n, xr_n, gl_n, gr_n, active_next)
        carry0 = (x, xl, xr, gl, gr, active)
        x_final, _, _, _, _, active_final = lax.fori_loop(0, int(maxiter), body, carry0)
        entropies = jax.vmap(entropy_func)(x_final)
        # Mark roots and entropies as nan where not valid
        roots_padded = jnp.where(valid, x_final, jnp.nan)
        entropies_padded = jnp.where(valid, entropies, jnp.nan)
        n_roots = jnp.sum(valid).astype(jnp.int32)
        # Best root: minimum entropy among valid
        entropy_masked = jnp.where(valid, entropies, jnp.inf)
        best_idx = jnp.argmin(entropy_masked)
        best_root = lax.cond(n_roots > 0, lambda _: x_final[best_idx], lambda _: jnp.asarray(0.0, dtype=jnp.float64), None)
        # --- DEBUG PRINTS ---
        jax.debug.print("[DEBUG/_roots] valid: {}", valid)
        jax.debug.print("[DEBUG/_roots] x_final: {}", x_final)
        jax.debug.print("[DEBUG/_roots] entropies: {}", entropies)
        jax.debug.print("[DEBUG/_roots] roots_padded: {}", roots_padded)
        jax.debug.print("[DEBUG/_roots] entropies_padded: {}", entropies_padded)
        jax.debug.print("[DEBUG/_roots] n_roots: {}", n_roots)
        jax.debug.print("[DEBUG/_roots] best_root: {}", best_root)
        return roots_padded, entropies_padded, best_root, n_roots

    roots, entropies, best_root, n_roots = lax.cond(n_brackets == 0, _empty, _roots)
    return roots, entropies, best_root, n_roots


# --- Vectorized profile root-finder ---
def find_all_ambipolar_Er_roots_profile_jit(
    get_Neoclassical_Fluxes,
    species,
    grid,
    field,
    database,
    state,
    er_min,
    er_max,
    n_scan=24,
    tol=1e-6,
    x_tol=1e-6,
    maxiter=12,
    blocksize=None,
):
    """
    Vectorized (vmap) root-finder for all radii. Returns roots, entropies, best_root for each radius.
    All arguments must be arrays or objects valid for all radii.
    """
    n_radial = state.Er.shape[0]

    def gamma_func(er_val, i):
        er_vec = state.Er.at[i].set(er_val)
        _, gamma_neo, _, _ = get_Neoclassical_Fluxes(
            species, grid, field, database, er_vec, state.temperature, state.density
        )
        return jnp.sum(species.charge_qp * gamma_neo[:, i])

    def entropy_func(er_val, i):
        er_vec = state.Er.at[i].set(er_val)
        _, gamma_neo, _, _ = get_Neoclassical_Fluxes(
            species, grid, field, database, er_vec, state.temperature, state.density
        )
        return jnp.sum(jnp.abs(gamma_neo[:, i]))

    def root_finder_for_radius(i):
        def gamma(er):
            return gamma_func(er, i)
        def entropy(er):
            return entropy_func(er, i)
        roots, entropies, best_root, n_roots = find_all_ambipolar_Er_roots_min_entropy_jit(
            gamma,
            entropy,
            Er_range=(er_min, er_max),
            n_scan=n_scan,
            tol=tol,
            x_tol=x_tol,
            maxiter=maxiter,
        )
        return roots, entropies, best_root, n_roots

    if blocksize is None:
        # Pure vmap over all radii
        roots_all, entropies_all, best_roots, n_roots_all = jax.vmap(root_finder_for_radius)(jnp.arange(n_radial))
    else:
        # Process in blocks
        n_blocks = (n_radial + blocksize - 1) // blocksize
        roots_list = []
        entropies_list = []
        best_roots_list = []
        n_roots_list = []
        for b in range(n_blocks):
            start = b * blocksize
            end = min((b + 1) * blocksize, n_radial)
            idxs = jnp.arange(start, end)
            roots_b, entropies_b, best_b, n_roots_b = jax.vmap(root_finder_for_radius)(idxs)
            roots_list.append(roots_b)
            entropies_list.append(entropies_b)
            best_roots_list.append(best_b)
            n_roots_list.append(n_roots_b)
        roots_all = jnp.concatenate(roots_list, axis=0)
        entropies_all = jnp.concatenate(entropies_list, axis=0)
        best_roots = jnp.concatenate(best_roots_list, axis=0)
        n_roots_all = jnp.concatenate(n_roots_list, axis=0)
    return roots_all, entropies_all, best_roots, n_roots_all


# --- Utility: Pad and sort up to 3 roots per radius for plotting ---
def pad_and_sort_roots_for_plotting(roots_all, entropies_all, n_roots_all, best_roots=None, max_roots=3):
    """
    For each radius, sort roots and entropies, pad to max_roots (default 3), and return arrays for plotting.
    roots_all: (n_radial, n_found) array
    entropies_all: (n_radial, n_found) array
    n_roots_all: (n_radial,) array (number of valid roots per radius)
    best_roots: (n_radial,) array (optional)
    Returns:
        roots_3: (max_roots, n_radial)
        entropies_3: (max_roots, n_radial)
        best_root: (n_radial,) if best_roots is not None else None
    """
    import numpy as np
    n_radial = roots_all.shape[0]
    roots_3 = np.full((max_roots, n_radial), np.nan)
    entropies_3 = np.full((max_roots, n_radial), np.nan)
    best_root_out = np.full(n_radial, np.nan) if best_roots is not None else None
    for i in range(n_radial):
        # Only select finite roots/entropies for this radius
        finite_mask = np.isfinite(roots_all[i])
        roots_np = np.array(roots_all[i][finite_mask])
        entropies_np = np.array(entropies_all[i][finite_mask])
        n_take = min(max_roots, len(roots_np))
        if n_take > 0:
            sort_idx = np.argsort(roots_np)
            roots_np = roots_np[sort_idx]
            entropies_np = entropies_np[sort_idx]
            roots_3[:n_take, i] = roots_np[:n_take]
            entropies_3[:n_take, i] = entropies_np[:n_take]
        if best_roots is not None and n_roots_all[i] > 0:
            best_root_out[i] = float(best_roots[i])
    return (roots_3, entropies_3, best_root_out) if best_roots is not None else (roots_3, entropies_3)