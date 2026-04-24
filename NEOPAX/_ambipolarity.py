import os
from pathlib import Path
import numpy as np
import h5py
import jax
import jax.numpy as jnp
from jax import lax

from ._transport_flux_models import build_transport_flux_model
from ._entropy_models import get_entropy_model


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
        x, gx, xl, xr, gl, gr, active = carry
        dgx = (gr - gl) / (xr - xl + 1e-12)
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
        return (x_trial, g_trial, xl_next, xr_next, gl_next, gr_next, active_next)

    carry0 = (x0, jax.vmap(Gamma_func)(x0), left, right, g_left, g_right, valid_init)
    x_final, _, _, _, _, _, active_final = lax.fori_loop(0, maxiter, body, carry0)

    # Keep initially valid candidates; non-converged values remain as best effort.
    valid_mask = valid_init
    entropy_all = jax.vmap(entropy_func)(x_final)
    entropy_masked = jnp.where(valid_mask, entropy_all, jnp.inf)

    best_idx = jnp.argmin(entropy_masked)
    has_any = jnp.any(valid_mask)
    best_root = jnp.where(has_any, x_final[best_idx], jnp.asarray(0.0, dtype=x_final.dtype))
    return best_root, x_final, entropy_masked, valid_mask



def find_ambipolar_Er_multistart_clustered(
    Gamma_func,
    entropy_func=None,
    Er_range=(-20.0, 20.0),
    n_starts=32,
    tol=1e-6,
    maxiter=30,
    cluster_tol=1e-3,
    return_entropies=False,
):
    """
    Multi-start root-finding for ambipolarity using least-squares and clustering.
    Uses many initial guesses, runs a root finder (optimistix if available), and clusters roots.
    Returns unique roots (within cluster_tol).
    """

    er_min, er_max = float(Er_range[0]), float(Er_range[1])
    starts = jnp.linspace(er_min, er_max, n_starts)

    def newton_step(x0):
        def body_fun(i, val):
            x, iter_idx = val
            gx = Gamma_func(x)
            dgx = jax.grad(Gamma_func)(x)
            x_new = x - gx / (dgx + 1e-12)
            x_new = jnp.clip(x_new, er_min, er_max)
            return (jnp.where(jnp.abs(gx) < tol, x, x_new), iter_idx + 1)
        x_init = x0
        i_init = 0
        x_final, _ = lax.fori_loop(0, maxiter, body_fun, (x_init, i_init))
        return x_final

    # Vectorized Newton from all starts
    roots_all = jax.vmap(newton_step)(starts)
    # Only keep roots that are finite and within bounds
    valid_mask = jnp.isfinite(roots_all) & (roots_all >= er_min) & (roots_all <= er_max)
    roots_valid = jnp.where(valid_mask, roots_all, jnp.nan)

    # Cluster roots: keep only unique roots within cluster_tol
    def cluster_unique(roots, tol):
        roots = jnp.sort(roots)
        mask = jnp.isfinite(roots)
        idxs = jnp.nonzero(mask, size=roots.shape[0], fill_value=0)[0]
        roots_finite = roots[idxs]
        n_finite = roots_finite.shape[0]
        def no_roots():
            return jnp.full_like(roots, jnp.nan)
        def some_roots():
            # Broadcasting-based uniqueness: for each root, compare to all previous
            diffs = jnp.abs(roots_finite[:, None] - roots_finite[None, :])
            # diffs[i, j] is |root_i - root_j|
            # For each i, set diffs[i, j] = inf for j >= i (ignore self and future)
            mask_upper = jnp.triu(jnp.ones_like(diffs), k=0)
            diffs_masked = jnp.where(mask_upper, jnp.inf, diffs)
            # A root is unique if all previous diffs > tol
            is_unique = jnp.all(diffs_masked > tol, axis=1)
            # Always keep the first root
            is_unique = is_unique.at[0].set(True)
            unique_roots = jnp.where(is_unique, roots_finite, jnp.nan)
            pad_len = roots.shape[0] - unique_roots.shape[0]
            unique_roots_padded = jnp.concatenate([unique_roots, jnp.full((pad_len,), jnp.nan)])
            return unique_roots_padded
        return lax.cond(n_finite == 0, no_roots, some_roots)

    unique_roots = cluster_unique(roots_valid, cluster_tol)

    if entropy_func is not None and return_entropies:
        entropies = jax.vmap(lambda er: entropy_func(er))(unique_roots)
        return unique_roots, entropies
    return unique_roots

# --- Two-stage adaptive root-finder for a single radius ---
def find_ambipolar_Er_min_entropy_jit_multires(
    Gamma_func,
    entropy_func,
    Er_range=(-20.0, 20.0),
    n_coarse=24,
    n_refine=8,
    max_roots=3,
    tol=1e-6,
    x_tol=1e-6,
    maxiter=12,
):
    """
    Two-stage (coarse/fine) JIT-friendly ambipolar root search for a single radius.
    1. Coarse scan to bracket roots (n_coarse)
    2. Refine each bracket with Newton-bisection (n_refine)
    3. Pad to max_roots for static shape
    """
    er_min = jnp.asarray(Er_range[0], dtype=jnp.float64)
    er_max = jnp.asarray(Er_range[1], dtype=jnp.float64)
    er_grid = jnp.linspace(er_min, er_max, n_coarse, dtype=jnp.float64)
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
    # --- JAX-friendly static-shape bracket gathering ---
    idxs = jnp.arange(left.shape[0])
    valid_idxs = jnp.where(valid, idxs, left.shape[0])  # invalids get out-of-bounds index
    sorted_valid_idxs = jnp.sort(valid_idxs)
    def safe_gather(arr):
        arr_ext = jnp.concatenate([arr, jnp.zeros((1,), arr.dtype)], axis=0)  # pad for out-of-bounds
        return arr_ext[sorted_valid_idxs[:max_roots]]

    xl = safe_gather(left)
    xr = safe_gather(right)
    gl = safe_gather(g_left)
    gr = safe_gather(g_right)
    n_found = jnp.minimum(jnp.sum(valid), max_roots)

    def _empty():
        roots = jnp.full((max_roots,), jnp.nan, dtype=jnp.float64)
        entropies = jnp.full((max_roots,), jnp.nan, dtype=jnp.float64)
        best_root = jnp.asarray(0.0, dtype=jnp.float64)
        n_roots = jnp.asarray(0, dtype=jnp.int32)
        return roots, entropies, best_root, n_roots

    def _roots():
        x = 0.5 * (xl + xr)
        active = jnp.arange(max_roots) < n_found
        def body(_, carry):
            x, gx, xl, xr, gl, gr, active = carry
            dgx = (gr - gl) / (xr - xl + 1e-12)
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
            return (x_trial, g_trial, xl_n, xr_n, gl_n, gr_n, active_next)
        carry0 = (x, jax.vmap(Gamma_func)(x), xl, xr, gl, gr, active)
        x_final, _, _, _, _, _, active_final = lax.fori_loop(0, n_refine, body, carry0)
        entropies = jax.vmap(entropy_func)(x_final)
        roots_padded = jnp.where(jnp.arange(max_roots) < n_found, x_final, jnp.nan)
        entropies_padded = jnp.where(jnp.arange(max_roots) < n_found, entropies, jnp.nan)
        n_roots = jnp.asarray(n_found, dtype=jnp.int32)
        entropy_masked = jnp.where(jnp.arange(max_roots) < n_found, entropies_padded, jnp.inf)
        best_idx = jnp.argmin(entropy_masked)
        best_root = lax.cond(n_found > 0, lambda _: roots_padded[best_idx], lambda _: jnp.asarray(0.0, dtype=jnp.float64), None)
        return roots_padded, entropies_padded, best_root, n_roots

    roots, entropies, best_root, n_roots = lax.cond(jnp.sum(valid) == 0, _empty, _roots)
    return roots, entropies, best_root, n_roots


def find_ambipolar_Er_min_entropy_jit_multires_continuation(
    Gamma_func,
    entropy_func,
    prev_root,
    global_Er_range=(-20.0, 20.0),
    local_half_width=6.0,
    expand_factor=2.0,
    max_local_expands=2,
    fallback_global=True,
    n_coarse=24,
    n_refine=8,
    max_roots=3,
    tol=1e-6,
    x_tol=1e-6,
    maxiter=12,
):
    """
    Continuation upgrade of the scalar two-stage search.

    Start from a narrow range around the previous radial best root and only
    widen/fallback if the local search fails. This preserves the JAX-friendly
    scalar solver while avoiding a broad global scan at every radius.
    """
    global_min = jnp.asarray(global_Er_range[0], dtype=jnp.float64)
    global_max = jnp.asarray(global_Er_range[1], dtype=jnp.float64)
    prev_root = jnp.asarray(prev_root, dtype=jnp.float64)

    def solve_range(er_min, er_max):
        return find_ambipolar_Er_min_entropy_jit_multires(
            Gamma_func=Gamma_func,
            entropy_func=entropy_func,
            Er_range=(er_min, er_max),
            n_coarse=n_coarse,
            n_refine=n_refine,
            max_roots=max_roots,
            tol=tol,
            x_tol=x_tol,
            maxiter=maxiter,
        )

    roots = jnp.full((max_roots,), jnp.nan, dtype=jnp.float64)
    entropies = jnp.full((max_roots,), jnp.nan, dtype=jnp.float64)
    best_root = jnp.asarray(prev_root, dtype=jnp.float64)
    n_roots = jnp.asarray(0, dtype=jnp.int32)
    found = jnp.asarray(False)

    for k in range(int(max_local_expands) + 1):
        half_width = jnp.asarray(local_half_width * (expand_factor ** k), dtype=jnp.float64)
        er_min = jnp.clip(prev_root - half_width, global_min, global_max)
        er_max = jnp.clip(prev_root + half_width, global_min, global_max)
        roots_k, entropies_k, best_root_k, n_roots_k = solve_range(er_min, er_max)
        use_k = (~found) & (n_roots_k > 0)
        roots = jnp.where(use_k, roots_k, roots)
        entropies = jnp.where(use_k, entropies_k, entropies)
        best_root = jnp.where(use_k, best_root_k, best_root)
        n_roots = jnp.where(use_k, n_roots_k, n_roots)
        found = found | (n_roots_k > 0)

    if fallback_global:
        roots_g, entropies_g, best_root_g, n_roots_g = solve_range(global_min, global_max)
        use_global = (~found) & (n_roots_g > 0)
        roots = jnp.where(use_global, roots_g, roots)
        entropies = jnp.where(use_global, entropies_g, entropies)
        best_root = jnp.where(use_global, best_root_g, best_root)
        n_roots = jnp.where(use_global, n_roots_g, n_roots)

    return roots, entropies, best_root, n_roots


def initialize_ambipolar_best_roots_fast(state, config, params, flux_model, amb_cfg):
    """
    Fast ambipolar best-root profile initializer.

    Optimized for initialization only:
    - robust global solve on the first radius
    - cheap local continuation search on later radii
    - optional fallback to global search only when local search fails

    Returns:
      best_roots: (n_radial,) ndarray
    """
    import dataclasses

    Er = getattr(state, "Er", None)
    if Er is None:
        raise ValueError("State must have an 'Er' attribute.")
    n_radial = Er.shape[0] if hasattr(Er, "shape") and len(Er.shape) == 1 else 1

    charge_qp = jnp.asarray(params["species"].charge_qp)
    local_particle_flux = flux_model.build_local_particle_flux_evaluator(state)

    def _evaluate_gamma_and_entropy(i, er):
        if local_particle_flux is not None:
            gamma = local_particle_flux(i, er)
        else:
            er_value = jnp.asarray(er, dtype=Er.dtype)
            if n_radial > 1:
                er_vec = Er.at[i].set(er_value)
            else:
                er_vec = jnp.asarray([er_value], dtype=Er.dtype)
            fluxes = flux_model(dataclasses.replace(state, Er=er_vec))
            gamma = fluxes.get("Gamma_total") or fluxes.get("Gamma")
            if gamma is None:
                raise ValueError("Flux model did not return 'Gamma' or 'Gamma_total'.")
            gamma = gamma[:, i]
        return (
            jnp.sum(charge_qp * gamma),
            jnp.sum(jnp.abs(gamma)),
        )

    def gamma_func_factory(i):
        def gamma(er):
            gamma_val, _ = _evaluate_gamma_and_entropy(i, er)
            return gamma_val
        return gamma

    def entropy_func_factory(i):
        def entropy(er):
            _, entropy_val = _evaluate_gamma_and_entropy(i, er)
            return entropy_val
        return entropy

    global_range = (
        float(amb_cfg.get("er_ambipolar_scan_min", -20.0)),
        float(amb_cfg.get("er_ambipolar_scan_max", 20.0)),
    )
    n_coarse = int(amb_cfg.get("er_ambipolar_n_coarse", 24))
    n_refine = int(amb_cfg.get("er_ambipolar_n_refine", 8))
    max_roots = int(amb_cfg.get("er_ambipolar_max_roots", 3))
    tol = float(amb_cfg.get("er_ambipolar_tol", 1e-6))
    x_tol = float(amb_cfg.get("er_ambipolar_x_tol", 1e-6))
    maxiter = int(amb_cfg.get("er_ambipolar_maxiter", 12))

    init_local_n_scan = int(amb_cfg.get("er_init_local_n_scan", 16))
    init_local_half_width = float(amb_cfg.get("er_init_local_half_width", 6.0))
    init_expand_factor = float(amb_cfg.get("er_init_expand_factor", 2.0))
    init_max_expands = int(amb_cfg.get("er_init_max_expands", 2))
    init_fallback_global = bool(amb_cfg.get("er_init_fallback_global", True))

    best_roots = np.full((n_radial,), np.nan, dtype=np.float64)

    def solve_global_best(i):
        _, _, best_root_i, n_roots_i = find_ambipolar_Er_min_entropy_jit_multires(
            Gamma_func=gamma_func_factory(i),
            entropy_func=entropy_func_factory(i),
            Er_range=global_range,
            n_coarse=n_coarse,
            n_refine=n_refine,
            max_roots=max_roots,
            tol=tol,
            x_tol=x_tol,
            maxiter=maxiter,
        )
        return float(np.asarray(best_root_i)), int(np.asarray(n_roots_i))

    def solve_local_best(i, prev_best, half_width):
        best_root_i, _, _, valid_mask = find_ambipolar_Er_min_entropy_jit(
            Gamma_func=gamma_func_factory(i),
            entropy_func=entropy_func_factory(i),
            Er_range=(
                max(global_range[0], prev_best - half_width),
                min(global_range[1], prev_best + half_width),
            ),
            n_scan=init_local_n_scan,
            tol=tol,
            x_tol=x_tol,
            maxiter=maxiter,
        )
        found = bool(np.asarray(jnp.any(valid_mask)))
        return float(np.asarray(best_root_i)), found

    prev_best = None
    for i in range(n_radial):
        if i == 0 or prev_best is None or not np.isfinite(prev_best):
            best_root_i, n_found = solve_global_best(i)
        else:
            found = False
            best_root_i = prev_best
            for k in range(init_max_expands + 1):
                half_width = init_local_half_width * (init_expand_factor ** k)
                candidate, found = solve_local_best(i, prev_best, half_width)
                if found and np.isfinite(candidate):
                    best_root_i = candidate
                    break
            if (not found) and init_fallback_global:
                best_root_i, _ = solve_global_best(i)
                found = np.isfinite(best_root_i)
            n_found = 1 if found else 0

        best_roots[i] = best_root_i
        if n_found > 0 and np.isfinite(best_root_i):
            prev_best = best_root_i

    return best_roots


def find_ambipolar_root_tracked_local(
    Gamma_func,
    prev_root,
    global_Er_range=(-20.0, 20.0),
    local_half_width=4.0,
    n_scan=12,
    tol=1e-6,
    x_tol=1e-6,
    maxiter=10,
):
    """
    Cheap single-root tracker around a previous best root.

    This is intended for initialization only:
    - scan a narrow local window
    - pick the bracket closest to the previous root
    - refine a single root with safeguarded Newton/bisection
    """
    global_min = jnp.asarray(global_Er_range[0], dtype=jnp.float64)
    global_max = jnp.asarray(global_Er_range[1], dtype=jnp.float64)
    prev_root = jnp.asarray(prev_root, dtype=jnp.float64)
    er_min = jnp.clip(prev_root - local_half_width, global_min, global_max)
    er_max = jnp.clip(prev_root + local_half_width, global_min, global_max)

    er_grid = jnp.linspace(er_min, er_max, n_scan, dtype=jnp.float64)
    gamma_grid = jax.vmap(Gamma_func)(er_grid)
    left = er_grid[:-1]
    right = er_grid[1:]
    g_left = gamma_grid[:-1]
    g_right = gamma_grid[1:]

    left_zero = jnp.abs(g_left) <= tol
    right_zero = jnp.abs(g_right) <= tol
    sign_change = (g_left * g_right) < 0.0
    valid = sign_change | left_zero | right_zero

    mids = 0.5 * (left + right)
    dist = jnp.abs(mids - prev_root)
    masked_dist = jnp.where(valid, dist, jnp.inf)
    best_idx = jnp.argmin(masked_dist)
    found = jnp.any(valid)

    xl = left[best_idx]
    xr = right[best_idx]
    gl = g_left[best_idx]
    gr = g_right[best_idx]
    x0 = jnp.where(left_zero[best_idx], xl, jnp.where(right_zero[best_idx], xr, 0.5 * (xl + xr)))

    def _refine(_):
        def body(_, carry):
            x, xl, xr, gl, gr, active = carry
            gx = Gamma_func(x)
            dgx = jax.grad(Gamma_func)(x)
            x_newton = x - gx / (dgx + 1e-12)
            x_bisect = 0.5 * (xl + xr)
            use_newton = active & jnp.isfinite(x_newton) & (x_newton > xl) & (x_newton < xr)
            x_trial = jnp.where(use_newton, x_newton, x_bisect)
            g_trial = Gamma_func(x_trial)
            same_sign_left = (gl * g_trial) > 0.0
            xl_n = jnp.where(active & same_sign_left, x_trial, xl)
            gl_n = jnp.where(active & same_sign_left, g_trial, gl)
            xr_n = jnp.where(active & ~same_sign_left, x_trial, xr)
            gr_n = jnp.where(active & ~same_sign_left, g_trial, gr)
            converged = active & ((jnp.abs(g_trial) <= tol) | (jnp.abs(xr_n - xl_n) <= x_tol))
            active_next = active & ~converged
            return (x_trial, xl_n, xr_n, gl_n, gr_n, active_next)

        carry0 = (x0, xl, xr, gl, gr, jnp.asarray(True))
        x_final, _, _, _, _, _ = lax.fori_loop(0, maxiter, body, carry0)
        return x_final

    root = lax.cond(found, _refine, lambda _: prev_root, operand=None)
    return root, found


def _build_initializer_evaluators(state, params, flux_model):
    import dataclasses

    Er = getattr(state, "Er", None)
    if Er is None:
        raise ValueError("State must have an 'Er' attribute.")
    n_radial = Er.shape[0] if hasattr(Er, "shape") and len(Er.shape) == 1 else 1

    charge_qp = jnp.asarray(params["species"].charge_qp)
    local_particle_flux = flux_model.build_local_particle_flux_evaluator(state)

    def _evaluate_gamma_and_entropy(i, er):
        if local_particle_flux is not None:
            gamma = local_particle_flux(i, er)
        else:
            er_value = jnp.asarray(er, dtype=Er.dtype)
            if n_radial > 1:
                er_vec = Er.at[i].set(er_value)
            else:
                er_vec = jnp.asarray([er_value], dtype=Er.dtype)
            fluxes = flux_model(dataclasses.replace(state, Er=er_vec))
            gamma = fluxes.get("Gamma_total") or fluxes.get("Gamma")
            if gamma is None:
                raise ValueError("Flux model did not return 'Gamma' or 'Gamma_total'.")
            gamma = gamma[:, i]
        return (
            jnp.sum(charge_qp * gamma),
            jnp.sum(jnp.abs(gamma)),
        )

    def gamma_func_factory(i):
        def gamma(er):
            gamma_val, _ = _evaluate_gamma_and_entropy(i, er)
            return gamma_val
        return gamma

    def entropy_func_factory(i):
        def entropy(er):
            _, entropy_val = _evaluate_gamma_and_entropy(i, er)
            return entropy_val
        return entropy

    return n_radial, gamma_func_factory, entropy_func_factory


def initialize_ambipolar_best_roots_tracked(state, config, params, flux_model, amb_cfg):
    """
    Faster initialization-oriented ambipolar best-root tracker.

    Strategy:
    - first radius: robust global two-stage + entropy selection
    - later radii: track the selected branch locally using a cheap sign-scan and
      single-root refinement
    - fallback to global two-stage only when local tracking fails
    """
    n_radial, gamma_func_factory, entropy_func_factory = _build_initializer_evaluators(
        state, params, flux_model
    )

    global_range = (
        float(amb_cfg.get("er_ambipolar_scan_min", -20.0)),
        float(amb_cfg.get("er_ambipolar_scan_max", 20.0)),
    )
    n_coarse = int(amb_cfg.get("er_ambipolar_n_coarse", 24))
    n_refine = int(amb_cfg.get("er_ambipolar_n_refine", 8))
    max_roots = int(amb_cfg.get("er_ambipolar_max_roots", 3))
    tol = float(amb_cfg.get("er_ambipolar_tol", 1e-6))
    x_tol = float(amb_cfg.get("er_ambipolar_x_tol", 1e-6))
    maxiter = int(amb_cfg.get("er_ambipolar_maxiter", 12))

    track_local_n_scan = int(amb_cfg.get("er_init_track_n_scan", 12))
    track_local_half_width = float(amb_cfg.get("er_init_track_half_width", 4.0))
    track_expand_factor = float(amb_cfg.get("er_init_track_expand_factor", 2.0))
    track_max_expands = int(amb_cfg.get("er_init_track_max_expands", 2))
    track_fallback_global = bool(amb_cfg.get("er_init_track_fallback_global", True))

    best_roots = np.full((n_radial,), np.nan, dtype=np.float64)

    def solve_global_best(i):
        _, _, best_root_i, n_roots_i = find_ambipolar_Er_min_entropy_jit_multires(
            Gamma_func=gamma_func_factory(i),
            entropy_func=entropy_func_factory(i),
            Er_range=global_range,
            n_coarse=n_coarse,
            n_refine=n_refine,
            max_roots=max_roots,
            tol=tol,
            x_tol=x_tol,
            maxiter=maxiter,
        )
        return float(np.asarray(best_root_i)), int(np.asarray(n_roots_i))

    prev_best = None
    for i in range(n_radial):
        if i == 0 or prev_best is None or not np.isfinite(prev_best):
            best_root_i, n_found = solve_global_best(i)
        else:
            found = False
            best_root_i = prev_best
            for k in range(track_max_expands + 1):
                half_width = track_local_half_width * (track_expand_factor ** k)
                candidate, found = find_ambipolar_root_tracked_local(
                    Gamma_func=gamma_func_factory(i),
                    prev_root=prev_best,
                    global_Er_range=global_range,
                    local_half_width=half_width,
                    n_scan=track_local_n_scan,
                    tol=tol,
                    x_tol=x_tol,
                    maxiter=maxiter,
                )
                candidate = float(np.asarray(candidate))
                found = bool(np.asarray(found))
                if found and np.isfinite(candidate):
                    best_root_i = candidate
                    break
            if (not found) and track_fallback_global:
                best_root_i, _ = solve_global_best(i)
                found = np.isfinite(best_root_i)
            n_found = 1 if found else 0

        best_roots[i] = best_root_i
        if n_found > 0 and np.isfinite(best_root_i):
            prev_best = best_root_i

    return best_roots


def initialize_ambipolar_best_roots_hybrid(state, config, params, flux_model, amb_cfg):
    """
    Hybrid initialization:
    - use cheap tracked local continuation most radii
    - periodically or when suspicious, revalidate with full two_stage

    This keeps the baseline two_stage logic in the loop, but avoids paying for
    it at every radius.
    """
    n_radial, gamma_func_factory, entropy_func_factory = _build_initializer_evaluators(
        state, params, flux_model
    )

    global_range = (
        float(amb_cfg.get("er_ambipolar_scan_min", -20.0)),
        float(amb_cfg.get("er_ambipolar_scan_max", 20.0)),
    )
    n_coarse = int(amb_cfg.get("er_ambipolar_n_coarse", 24))
    n_refine = int(amb_cfg.get("er_ambipolar_n_refine", 8))
    max_roots = int(amb_cfg.get("er_ambipolar_max_roots", 3))
    tol = float(amb_cfg.get("er_ambipolar_tol", 1e-6))
    x_tol = float(amb_cfg.get("er_ambipolar_x_tol", 1e-6))
    maxiter = int(amb_cfg.get("er_ambipolar_maxiter", 12))

    track_local_n_scan = int(amb_cfg.get("er_init_track_n_scan", 12))
    track_local_half_width = float(amb_cfg.get("er_init_track_half_width", 4.0))
    track_expand_factor = float(amb_cfg.get("er_init_track_expand_factor", 2.0))
    track_max_expands = int(amb_cfg.get("er_init_track_max_expands", 2))
    track_fallback_global = bool(amb_cfg.get("er_init_track_fallback_global", True))
    revalidate_every = int(amb_cfg.get("er_init_hybrid_revalidate_every", 8))
    jump_trigger = float(amb_cfg.get("er_init_hybrid_jump_trigger", 8.0))

    best_roots = np.full((n_radial,), np.nan, dtype=np.float64)

    def solve_global_best(i):
        _, _, best_root_i, n_roots_i = find_ambipolar_Er_min_entropy_jit_multires(
            Gamma_func=gamma_func_factory(i),
            entropy_func=entropy_func_factory(i),
            Er_range=global_range,
            n_coarse=n_coarse,
            n_refine=n_refine,
            max_roots=max_roots,
            tol=tol,
            x_tol=x_tol,
            maxiter=maxiter,
        )
        return float(np.asarray(best_root_i)), int(np.asarray(n_roots_i))

    prev_best = None
    for i in range(n_radial):
        must_global = (
            i == 0
            or prev_best is None
            or not np.isfinite(prev_best)
            or (revalidate_every > 0 and (i % revalidate_every) == 0)
        )

        if must_global:
            best_root_i, n_found = solve_global_best(i)
        else:
            found = False
            best_root_i = prev_best
            for k in range(track_max_expands + 1):
                half_width = track_local_half_width * (track_expand_factor ** k)
                candidate, found = find_ambipolar_root_tracked_local(
                    Gamma_func=gamma_func_factory(i),
                    prev_root=prev_best,
                    global_Er_range=global_range,
                    local_half_width=half_width,
                    n_scan=track_local_n_scan,
                    tol=tol,
                    x_tol=x_tol,
                    maxiter=maxiter,
                )
                candidate = float(np.asarray(candidate))
                found = bool(np.asarray(found))
                if found and np.isfinite(candidate):
                    best_root_i = candidate
                    break

            suspicious = (not found) or (np.isfinite(best_root_i) and np.abs(best_root_i - prev_best) > jump_trigger)
            if suspicious and track_fallback_global:
                best_root_i, _ = solve_global_best(i)
                found = np.isfinite(best_root_i)
            n_found = 1 if found else 0

        best_roots[i] = best_root_i
        if n_found > 0 and np.isfinite(best_root_i):
            prev_best = best_root_i

    return best_roots


def _deduplicate_sorted_roots(roots, merge_tol):
    roots = np.asarray(roots, dtype=float)
    roots = roots[np.isfinite(roots)]
    if roots.size == 0:
        return roots
    roots = np.sort(roots)
    unique = [roots[0]]
    for x in roots[1:]:
        if abs(x - unique[-1]) > merge_tol:
            unique.append(x)
    return np.asarray(unique, dtype=float)


def initialize_ambipolar_best_roots_multibranch(state, config, params, flux_model, amb_cfg):
    """
    Multibranch continuation initializer.

    - first radius: full two_stage to get all candidate roots
    - later radii: track each previously found branch locally
    - choose minimum-entropy among tracked branches
    - periodically or on failure, fall back to full two_stage
    """
    n_radial, gamma_func_factory, entropy_func_factory = _build_initializer_evaluators(
        state, params, flux_model
    )

    global_range = (
        float(amb_cfg.get("er_ambipolar_scan_min", -20.0)),
        float(amb_cfg.get("er_ambipolar_scan_max", 20.0)),
    )
    n_coarse = int(amb_cfg.get("er_ambipolar_n_coarse", 24))
    n_refine = int(amb_cfg.get("er_ambipolar_n_refine", 8))
    max_roots = int(amb_cfg.get("er_ambipolar_max_roots", 3))
    tol = float(amb_cfg.get("er_ambipolar_tol", 1e-6))
    x_tol = float(amb_cfg.get("er_ambipolar_x_tol", 1e-6))
    maxiter = int(amb_cfg.get("er_ambipolar_maxiter", 12))

    track_local_n_scan = int(amb_cfg.get("er_init_track_n_scan", 12))
    track_local_half_width = float(amb_cfg.get("er_init_track_half_width", 4.0))
    track_expand_factor = float(amb_cfg.get("er_init_track_expand_factor", 2.0))
    track_max_expands = int(amb_cfg.get("er_init_track_max_expands", 2))
    track_fallback_global = bool(amb_cfg.get("er_init_track_fallback_global", True))
    revalidate_every = int(amb_cfg.get("er_init_multibranch_revalidate_every", 8))
    merge_tol = float(amb_cfg.get("er_init_multibranch_merge_tol", 1e-2))

    best_roots = np.full((n_radial,), np.nan, dtype=np.float64)

    def solve_global_set(i):
        roots_i, entropies_i, best_root_i, n_roots_i = find_ambipolar_Er_min_entropy_jit_multires(
            Gamma_func=gamma_func_factory(i),
            entropy_func=entropy_func_factory(i),
            Er_range=global_range,
            n_coarse=n_coarse,
            n_refine=n_refine,
            max_roots=max_roots,
            tol=tol,
            x_tol=x_tol,
            maxiter=maxiter,
        )
        roots_np = np.asarray(roots_i, dtype=float)
        entropies_np = np.asarray(entropies_i, dtype=float)
        valid = np.isfinite(roots_np)
        return roots_np[valid], entropies_np[valid], float(np.asarray(best_root_i)), int(np.asarray(n_roots_i))

    prev_roots = None
    for i in range(n_radial):
        must_global = (
            i == 0
            or prev_roots is None
            or len(prev_roots) == 0
            or (revalidate_every > 0 and (i % revalidate_every) == 0)
        )

        if must_global:
            roots_np, entropies_np, best_root_i, n_found = solve_global_set(i)
            prev_roots = roots_np
        else:
            candidates = []
            for root_prev in prev_roots:
                found = False
                candidate = root_prev
                for k in range(track_max_expands + 1):
                    half_width = track_local_half_width * (track_expand_factor ** k)
                    candidate_jax, found = find_ambipolar_root_tracked_local(
                        Gamma_func=gamma_func_factory(i),
                        prev_root=root_prev,
                        global_Er_range=global_range,
                        local_half_width=half_width,
                        n_scan=track_local_n_scan,
                        tol=tol,
                        x_tol=x_tol,
                        maxiter=maxiter,
                    )
                    candidate = float(np.asarray(candidate_jax))
                    found = bool(np.asarray(found))
                    if found and np.isfinite(candidate):
                        candidates.append(candidate)
                        break

            candidates = _deduplicate_sorted_roots(candidates, merge_tol)
            if candidates.size > 0:
                entropies_np = np.asarray([float(np.asarray(entropy_func_factory(i)(r))) for r in candidates], dtype=float)
                best_root_i = float(candidates[int(np.argmin(entropies_np))])
                n_found = int(candidates.size)
                prev_roots = candidates
            elif track_fallback_global:
                roots_np, entropies_np, best_root_i, n_found = solve_global_set(i)
                prev_roots = roots_np
            else:
                best_root_i = np.nan if prev_roots is None or len(prev_roots) == 0 else float(prev_roots[0])
                n_found = 0

        best_roots[i] = best_root_i

    return best_roots

# --- Adaptive bracketing root-finder ---

def find_ambipolar_Er_min_entropy_jit_adaptive(
    Gamma_func,
    entropy_func,
    Er_range=(-20.0, 20.0),
    n_init=16,
    n_subdiv=2,
    n_rounds=2,
    max_brackets=24,
    n_refine=8,
    max_roots=3,
    tol=1e-6,
    x_tol=1e-6,
    maxiter=12,
):
    """
    JAX-compatible adaptive bracketing root-finder with static padding.
    - n_init: initial number of coarse intervals
    - n_subdiv: number of subdivisions per round (e.g., 2 for bisection)
    - n_rounds: number of adaptive subdivision rounds
    - max_brackets: maximum number of brackets to pad for JIT
    - n_refine: Newton-bisection steps per bracket
    - max_roots: maximum roots to pad for output
    """
    er_min, er_max = float(Er_range[0]), float(Er_range[1])
    # Initial coarse grid
    er_grid = jnp.linspace(er_min, er_max, n_init, dtype=jnp.float64)
    lefts = er_grid[:-1]
    rights = er_grid[1:]
    brackets = jnp.stack([lefts, rights], axis=1)  # shape (n_init-1, 2)
    n_brackets = brackets.shape[0]

    def bracket_signs(brackets):
        l, r = brackets[:, 0], brackets[:, 1]
        g_l = jax.vmap(Gamma_func)(l)
        g_r = jax.vmap(Gamma_func)(r)
        left_zero = jnp.abs(g_l) <= tol
        right_zero = jnp.abs(g_r) <= tol
        sign_change = (g_l * g_r) < 0.0
        valid = sign_change | left_zero | right_zero
        return valid

    # Adaptive subdivision rounds
    for _ in range(n_rounds):
        valid = bracket_signs(brackets)
        # Only subdivide valid brackets
        l, r = brackets[:, 0], brackets[:, 1]
        # Subdivide each valid bracket into n_subdiv sub-intervals
        def subdivide_bracket(lr):
            l, r = lr[0], lr[1]
            return jnp.linspace(l, r, n_subdiv + 1, dtype=jnp.float64)
        # For each valid bracket, get sub-interval edges
        sub_edges = jax.vmap(subdivide_bracket)(brackets)
        # Flatten all sub-intervals
        sub_lefts = sub_edges[:, :-1].reshape(-1)
        sub_rights = sub_edges[:, 1:].reshape(-1)
        # Only keep sub-intervals from valid brackets
        sub_lefts = sub_lefts[jnp.repeat(valid, n_subdiv)]
        sub_rights = sub_rights[jnp.repeat(valid, n_subdiv)]
        # Stack for next round
        brackets = jnp.stack([sub_lefts, sub_rights], axis=1)
        # Pad to max_brackets for JIT
        pad_amt = max_brackets - brackets.shape[0]
        if pad_amt > 0:
            pad_brackets = jnp.full((pad_amt, 2), jnp.nan)
            brackets = jnp.concatenate([brackets, pad_brackets], axis=0)
        brackets = brackets[:max_brackets]

    # Final valid brackets
    valid = bracket_signs(brackets)
    l, r = brackets[:, 0], brackets[:, 1]
    n_found = jnp.sum(valid)
    # Pad to max_brackets for JIT
    l = jnp.where(valid, l, jnp.nan)
    r = jnp.where(valid, r, jnp.nan)

    # Refine each bracket
    x = 0.5 * (l + r)
    active = jnp.isfinite(x)
    def body(_, carry):
        x, l, r, active = carry
        gx = jax.vmap(Gamma_func)(x)
        dgx = jax.vmap(jax.grad(Gamma_func))(x)
        x_newton = x - gx / (dgx + 1e-12)
        x_bisect = 0.5 * (l + r)
        use_newton = active & jnp.isfinite(x_newton) & (x_newton > l) & (x_newton < r)
        x_trial = jnp.where(use_newton, x_newton, x_bisect)
        g_trial = jax.vmap(Gamma_func)(x_trial)
        same_sign_left = (jax.vmap(Gamma_func)(l) * g_trial) > 0.0
        l_n = jnp.where(active & same_sign_left, x_trial, l)
        r_n = jnp.where(active & ~same_sign_left, x_trial, r)
        converged = active & ((jnp.abs(g_trial) <= tol) | (jnp.abs(r_n - l_n) <= x_tol))
        active_next = active & ~converged
        return (x_trial, l_n, r_n, active_next)
    carry0 = (x, l, r, active)
    x_final, _, _, active_final = lax.fori_loop(0, n_refine, body, carry0)
    entropies = jax.vmap(entropy_func)(x_final)
    # Only keep finite roots
    roots_padded = jnp.where(jnp.isfinite(x_final), x_final, jnp.nan)
    entropies_padded = jnp.where(jnp.isfinite(x_final), entropies, jnp.nan)
    # Pad to max_roots for output
    idxs = jnp.argsort(roots_padded)
    roots_sorted = roots_padded[idxs][:max_roots]
    entropies_sorted = entropies_padded[idxs][:max_roots]
    n_roots = jnp.asarray(jnp.sum(jnp.isfinite(roots_sorted)), dtype=jnp.int32)
    entropy_masked = jnp.where(jnp.isfinite(entropies_sorted), entropies_sorted, jnp.inf)
    best_idx = jnp.argmin(entropy_masked)
    best_root = lax.cond(n_roots > 0, lambda _: roots_sorted[best_idx], lambda _: jnp.asarray(0.0, dtype=jnp.float64), None)
    return roots_sorted, entropies_sorted, best_root, n_roots


# --- Vectorized profile root-finder ---





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



def plot_roots(
    rho,
    roots_3,
    entropies_3,
    best_root,
    output_dir: Path,
    overlay_reference_er: bool = True,
    reference_er_file: str | Path | None = None,
    reference_er_label: str | None = None,
) -> Path:
    # --- DEBUG: Print root/entropy arrays before plotting ---
    import numpy as np
    print("[DEBUG] roots_3 shape:", np.shape(roots_3))
    print("[DEBUG] roots_3 (first 5 radii):", np.array(roots_3)[:, :5])
    print("[DEBUG] entropies_3 shape:", np.shape(entropies_3))
    print("[DEBUG] entropies_3 (first 5 radii):", np.array(entropies_3)[:, :5])
    print("[DEBUG] best_root shape:", np.shape(best_root))
    print("[DEBUG] best_root (first 5):", np.array(best_root)[:5])

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required to plot ambipolar roots.") from exc

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 8.0), sharex=True)

    labels = ["root 1", "root 2", "root 3"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    # --- top panel: Er roots ---
    for k in range(3):
        ax1.plot(rho, roots_3[k], color=colors[k], linewidth=1.8, label=labels[k])
    ax1.plot(rho, best_root, color="black", linewidth=2.2, linestyle="--", label="min-entropy root")

    if overlay_reference_er:
        er_file = Path(reference_er_file) if reference_er_file is not None else Path("./examples/inputs/NTSS_Initial_Er_Opt.h5")
        er_label = reference_er_label or "reference Er"
        try:
            import h5py

            if er_file.is_file():
                with h5py.File(er_file, "r") as f:
                    r_data = np.asarray(f["r"][()])
                    er_data = np.asarray(f["Er"][()])
                    if r_data.ndim != 1 or er_data.ndim != 1 or r_data.size == 0 or er_data.size == 0:
                        raise ValueError("reference Er file must contain 1D non-empty 'r' and 'Er' datasets")
                    rho_ref = r_data / max(float(r_data[-1]), 1.0e-12)
                    er_interp = np.interp(np.asarray(rho), rho_ref, er_data)
                    ax1.plot(
                        rho,
                        er_interp,
                        color="tab:red",
                        linewidth=2.0,
                        linestyle=":",
                        label=er_label,
                    )
        except Exception as e:
            print(f"Could not plot ambipolarity reference Er from {er_file}: {e}")

    ax1.set_ylabel(r"$E_r$")
    ax1.set_title("Ambipolar $E_r$ roots (up to three per radius)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    # --- bottom panel: entropy at each root branch ---
    for k in range(3):
        ax2.plot(rho, entropies_3[k], color=colors[k], linewidth=1.8, label=labels[k])
    ax2.set_xlabel(r"$\rho$")
    ax2.set_ylabel(r"$\sum_s |\Gamma_s|$ (entropy proxy)")
    ax2.set_title("Entropy proxy per root branch")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    out_png = Path(output_dir) / "Er_ambipolar_roots.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    return out_png

def write_ambipolarity_hdf5(rho, roots_3, entropies_3, best_root, output_dir: Path) -> Path:
    out_h5 = Path(output_dir) / "Er_ambipolar_roots.h5"
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("rho", data=np.asarray(rho))
        f.create_dataset("roots_3", data=np.asarray(roots_3))
        f.create_dataset("entropies_3", data=np.asarray(entropies_3))
        f.create_dataset("best_root", data=np.asarray(best_root))
    return out_h5


# --- Ambipolarity Model Registry and Config-Driven Entrypoint ---
AMBIPOLARITY_MODEL_REGISTRY = {}

def register_ambipolarity_model(name: str, func):
    AMBIPOLARITY_MODEL_REGISTRY[str(name).strip().lower()] = func


# --- JIT/differentiable radial root-finder with blocksize option ---
def solve_ambipolarity_roots_radial(state, config, params, model_name, flux_model, entropy_model, amb_cfg):
    """
    JIT/differentiable radial root-finder with blocksize option (default: pure vmap).
    """
    import dataclasses

    Er = getattr(state, "Er", None)
    if Er is None:
        raise ValueError("State must have an 'Er' attribute.")
    n_radial = Er.shape[0] if hasattr(Er, "shape") and len(Er.shape) == 1 else 1
    debug_stage_markers = bool(params.get("solver_parameters", {}).get("debug_stage_markers", False))

    # Read er_ambipolar_blocksize from [ambipolarity] config (0 or unset = auto/all)
    er_ambipolar_blocksize = amb_cfg.get("er_ambipolar_blocksize", None)
    if er_ambipolar_blocksize is not None:
        try:
            er_ambipolar_blocksize = int(er_ambipolar_blocksize)
        except Exception:
            raise ValueError("[ambipolarity].er_ambipolar_blocksize must be an integer if specified.")
        if er_ambipolar_blocksize == 0:
            er_ambipolar_blocksize = None
    # If not set, treat as None (auto/all)
    else:
        er_ambipolar_blocksize = None

    charge_qp = jnp.asarray(params["species"].charge_qp)
    t_flux_build = __import__("time").perf_counter() if debug_stage_markers else None
    local_particle_flux = flux_model.build_local_particle_flux_evaluator(state)
    if debug_stage_markers and model_name == "two_stage":
        dt_flux_build = __import__("time").perf_counter() - t_flux_build
        print(
            "[NEOPAX] ambipolar two_stage setup: "
            f"local_flux_builder_elapsed_s={dt_flux_build:.3f} "
            f"uses_local_evaluator={local_particle_flux is not None}"
        )

    def _evaluate_gamma_and_entropy(i, er):
        if local_particle_flux is not None:
            gamma = local_particle_flux(i, er)
        else:
            er_value = jnp.asarray(er, dtype=Er.dtype)
            if n_radial > 1:
                er_vec = Er.at[i].set(er_value)
            else:
                er_vec = jnp.asarray([er_value], dtype=Er.dtype)
            fluxes = flux_model(dataclasses.replace(state, Er=er_vec))
            gamma = fluxes.get("Gamma_total") or fluxes.get("Gamma")
            if gamma is None:
                raise ValueError("Flux model did not return 'Gamma' or 'Gamma_total'.")
            gamma = gamma[:, i]
        return (
            jnp.sum(charge_qp * gamma),
            jnp.sum(jnp.abs(gamma)),
        )

    def gamma_func_factory(i):
        def gamma(er):
            gamma_val, _ = _evaluate_gamma_and_entropy(i, er)
            return gamma_val
        return gamma

    def entropy_func_factory(i):
        def entropy(er):
            _, entropy_val = _evaluate_gamma_and_entropy(i, er)
            return entropy_val
        return entropy

    root_finder = AMBIPOLARITY_MODEL_REGISTRY.get(model_name)
    if root_finder is None:
        raise ValueError(f"Unknown ambipolarity model: {model_name}")

    def root_finder_for_radius(i):
        args = {}
        if model_name == "two_stage":
            args.update({
                'Er_range': (
                    float(amb_cfg.get("er_ambipolar_scan_min", -20.0)),
                    float(amb_cfg.get("er_ambipolar_scan_max", 20.0)),
                ),
                'n_coarse': int(amb_cfg.get("er_ambipolar_n_coarse", 24)),
                'n_refine': int(amb_cfg.get("er_ambipolar_n_refine", 8)),
                'max_roots': int(amb_cfg.get("er_ambipolar_max_roots", 3)),
                'tol': float(amb_cfg.get("er_ambipolar_tol", 1e-6)),
                'x_tol': float(amb_cfg.get("er_ambipolar_x_tol", 1e-6)),
                'maxiter': int(amb_cfg.get("er_ambipolar_maxiter", 12)),
            })
        elif model_name == "adaptive":
            args.update({
                'Er_range': (
                    float(amb_cfg.get("er_ambipolar_scan_min", -20.0)),
                    float(amb_cfg.get("er_ambipolar_scan_max", 20.0)),
                ),
                'n_init': int(amb_cfg.get("er_ambipolar_adaptive_n_init", 16)),
                'n_subdiv': int(amb_cfg.get("er_ambipolar_adaptive_n_subdiv", 2)),
                'n_rounds': int(amb_cfg.get("er_ambipolar_adaptive_n_rounds", 2)),
                'max_brackets': int(amb_cfg.get("er_ambipolar_adaptive_max_brackets", 24)),
                'n_refine': int(amb_cfg.get("er_ambipolar_n_refine", 8)),
                'max_roots': int(amb_cfg.get("er_ambipolar_max_roots", 3)),
                'tol': float(amb_cfg.get("er_ambipolar_tol", 1e-6)),
                'x_tol': float(amb_cfg.get("er_ambipolar_x_tol", 1e-6)),
                'maxiter': int(amb_cfg.get("er_ambipolar_maxiter", 12)),
            })
        elif model_name in ("multistart", "multistart_clustered"):
            args.update({
                'Er_range': (
                    float(amb_cfg.get("er_ambipolar_scan_min", -20.0)),
                    float(amb_cfg.get("er_ambipolar_scan_max", 20.0)),
                ),
                'n_starts': int(amb_cfg.get("er_ambipolar_n_starts", 32)),
                'tol': float(amb_cfg.get("er_ambipolar_tol", 1e-6)),
                'maxiter': int(amb_cfg.get("er_ambipolar_maxiter", 12)),
                'cluster_tol': float(amb_cfg.get("er_ambipolar_cluster_tol", 1e-3)),
            })
        else:
            raise ValueError(f"Ambipolarity model '{model_name}' not recognized or not implemented.")
        args['Gamma_func'] = gamma_func_factory(i)
        args['entropy_func'] = entropy_func_factory(i)
        return root_finder(**args)

    batched_root_finder = jax.vmap(root_finder_for_radius)

    # Pure vmap or blocked vmap. Dispatch here so each mode keeps its own
    # smaller compiled program.
    if er_ambipolar_blocksize is None or er_ambipolar_blocksize >= n_radial:
        # JIT the full radial batch once when we can fit it in memory.
        if debug_stage_markers and model_name == "two_stage":
            print(f"[NEOPAX] ambipolar two_stage radial solve: mode=full_vmap n_radial={n_radial}")
        full_eval = jax.jit(batched_root_finder)
        t_eval = __import__("time").perf_counter() if debug_stage_markers and model_name == "two_stage" else None
        result = full_eval(jnp.arange(n_radial, dtype=jnp.int32))
        if debug_stage_markers and model_name == "two_stage":
            dt_eval = __import__("time").perf_counter() - t_eval
            print(
                f"[NEOPAX] ambipolar two_stage full_vmap elapsed_s={dt_eval:.3f}"
            )
        return tuple(np.asarray(arr)[:n_radial] for arr in result)
    else:
        # Blocked vmap for memory efficiency.
        block_size = er_ambipolar_blocksize
        n_blocks = (n_radial + block_size - 1) // block_size
        if debug_stage_markers and model_name == "two_stage":
            print(
                f"[NEOPAX] ambipolar two_stage radial solve: mode=blocked "
                f"n_radial={n_radial} block_size={block_size} n_blocks={n_blocks}"
            )
        if model_name in ("two_stage", "adaptive"):
            max_roots = int(amb_cfg.get("er_ambipolar_max_roots", 3))
            roots_shape = (n_radial, max_roots)
            entropies_shape = (n_radial, max_roots)
            best_shape = (n_radial,)
            n_roots_shape = (n_radial,)
            roots_dtype = jnp.float64
            entropies_dtype = jnp.float64
            best_dtype = jnp.float64
        else:
            idxs0 = jnp.arange(block_size)
            roots0, entropies0, best0, n_roots0 = batched_root_finder(idxs0)
            roots_shape = (n_radial,) + roots0.shape[1:]
            entropies_shape = (n_radial,) + entropies0.shape[1:]
            best_shape = (n_radial,) if best0.shape == () else (n_radial,) + best0.shape[1:]
            n_roots_shape = (n_radial,) + n_roots0.shape[1:]
            roots_dtype = roots0.dtype
            entropies_dtype = entropies0.dtype
            best_dtype = best0.dtype

        if model_name in ("two_stage", "adaptive"):
            def evaluate_block(idxs):
                clipped = jnp.clip(idxs, 0, n_radial - 1)
                valid = idxs < n_radial
                roots_b, entropies_b, best_b, n_roots_b = batched_root_finder(clipped)
                roots_b = jnp.where(valid[:, None], roots_b, jnp.nan)
                entropies_b = jnp.where(valid[:, None], entropies_b, jnp.nan)
                best_b = jnp.where(valid, best_b, jnp.nan)
                n_roots_b = jnp.where(valid, n_roots_b, 0)
                return roots_b, entropies_b, best_b, n_roots_b

            block_eval = jax.jit(evaluate_block)
        else:
            block_eval = jax.jit(batched_root_finder)

        def init_arrays():
            nan_roots = jnp.full(roots_shape, jnp.nan, dtype=roots_dtype)
            nan_entropies = jnp.full(entropies_shape, jnp.nan, dtype=entropies_dtype)
            nan_best = jnp.full(best_shape, jnp.nan, dtype=best_dtype)
            zero_n_roots = jnp.zeros(n_roots_shape, dtype=jnp.int32)
            return (
                nan_roots,
                nan_entropies,
                nan_best,
                zero_n_roots,
            )

        def body_fun(b, carry):
            roots_all, entropies_all, best_roots, n_roots_all = carry
            start = b * block_size
            stop = min(start + block_size, n_radial)
            n_valid = stop - start
            idxs = jnp.arange(start, start + block_size, dtype=jnp.int32)
            roots_b, entropies_b, best_b, n_roots_b = block_eval(idxs)
            if roots_b.ndim == 1:
                roots_all = roots_all.at[start:stop].set(roots_b[:n_valid])
                entropies_all = entropies_all.at[start:stop].set(entropies_b[:n_valid])
                best_roots = best_roots.at[start:stop].set(best_b[:n_valid])
                n_roots_all = n_roots_all.at[start:stop].set(n_roots_b[:n_valid])
            else:
                roots_all = roots_all.at[start:stop, ...].set(roots_b[:n_valid])
                entropies_all = entropies_all.at[start:stop, ...].set(entropies_b[:n_valid])
                best_roots = best_roots.at[start:stop, ...].set(best_b[:n_valid])
                n_roots_all = n_roots_all.at[start:stop, ...].set(n_roots_b[:n_valid])
            return (roots_all, entropies_all, best_roots, n_roots_all)

        roots_all, entropies_all, best_roots, n_roots_all = init_arrays()
        for b in range(n_blocks):
            if debug_stage_markers and model_name == "two_stage":
                t_block = __import__("time").perf_counter()
            roots_all, entropies_all, best_roots, n_roots_all = body_fun(
                b, (roots_all, entropies_all, best_roots, n_roots_all)
            )
            if debug_stage_markers and model_name == "two_stage":
                dt_block = __import__("time").perf_counter() - t_block
                start = b * block_size
                stop = min(start + block_size, n_radial)
                print(
                    f"[NEOPAX] ambipolar two_stage block {b+1}/{n_blocks}: "
                    f"indices={start}:{stop} elapsed_s={dt_block:.3f}"
                )
                if b == 0:
                    print(
                        "[NEOPAX] ambipolar two_stage memory knobs: "
                        f"n_coarse={int(amb_cfg.get('er_ambipolar_n_coarse', 24))} "
                        f"n_refine={int(amb_cfg.get('er_ambipolar_n_refine', 8))} "
                        f"max_roots={int(amb_cfg.get('er_ambipolar_max_roots', 3))} "
                        f"block_size={block_size}"
                    )
        return tuple(np.asarray(arr)[:n_radial] for arr in (roots_all, entropies_all, best_roots, n_roots_all))

# --- Orchestration/config reading only ---
def solve_ambipolarity_roots_from_config(state, config, params, flux_model=None, entropy_model=None):
    """
    Config-driven entrypoint for ambipolarity root finding. Reads config, builds models, and calls the JIT/diff radial solver.
    """
    general = config.get("general", {})
    mode = general.get("mode", config.get("mode", "transport")).lower()
    if mode != "ambipolarity":
        raise RuntimeError("solve_ambipolarity_roots_from_config called with mode != 'ambipolarity'")

    amb_cfg = config.get("ambipolarity", {})
    neoclassical_cfg = config.get("neoclassical", {})

    # Select root-finding model
    model_name = amb_cfg.get("er_ambipolar_method", "two_stage").lower()

    if flux_model is None:
        from ._transport_flux_models import ZeroTransportModel, get_transport_flux_model

        neoclassical_factory = get_transport_flux_model(config.get("neoclassical", {}).get("flux_model", "monkes_database"))
        turbulence_factory = get_transport_flux_model(config.get("turbulence", {}).get("flux_model", "none"))
        classical_factory = get_transport_flux_model(config.get("classical", {}).get("flux_model", "none")) if "classical" in config else None

        species = params["species"]
        energy_grid = params["energy_grid"]
        geometry = params["geometry"]
        database = params["database"]

        neoclassical_model = neoclassical_factory(species, energy_grid, geometry, database)
        turbulence_model = turbulence_factory(species, energy_grid, geometry, database) if turbulence_factory is not None else ZeroTransportModel()
        classical_model = classical_factory(species, energy_grid, geometry, database) if classical_factory is not None else ZeroTransportModel()
        flux_model = build_transport_flux_model(neoclassical_model, turbulence_model, classical_model)

    if entropy_model is None:
        entropy_model_name = neoclassical_cfg.get(
            "entropy_model",
            params["solver_parameters"].get("neoclassical_flux_model", "monkes_database"),
        )
        entropy_model = get_entropy_model(entropy_model_name)

    # Call the JIT/diff radial root-finder
    roots_all, entropies_all, best_roots, n_roots_all = solve_ambipolarity_roots_radial(
        state, config, params, model_name, flux_model, entropy_model, amb_cfg
    )

    # Read plotting/output options from config
    er_ambipolar_plot = amb_cfg.get("er_ambipolar_plot", False)
    er_ambipolar_write_hdf5 = amb_cfg.get("er_ambipolar_write_hdf5", False)
    er_ambipolar_output_dir = amb_cfg.get("er_ambipolar_output_dir", None)

    # Ensure only the first n_radial valid entries are used (in case of blocksize padding)
    n_radial = getattr(state, "Er", None)
    if n_radial is not None and hasattr(n_radial, "shape"):
        n_radial = n_radial.shape[0]
    else:
        n_radial = best_roots.shape[0] if hasattr(best_roots, "shape") else 1

    roots_all = np.asarray(roots_all)[:n_radial]
    entropies_all = np.asarray(entropies_all)[:n_radial]
    best_roots = np.asarray(best_roots)[:n_radial]
    n_roots_all = np.asarray(n_roots_all)[:n_radial]

    return roots_all, entropies_all, best_roots, n_roots_all, er_ambipolar_plot, er_ambipolar_write_hdf5, er_ambipolar_output_dir


# Register the numerical methods as ambipolarity models

register_ambipolarity_model("two_stage", find_ambipolar_Er_min_entropy_jit_multires)

register_ambipolarity_model("adaptive", find_ambipolar_Er_min_entropy_jit_adaptive)
