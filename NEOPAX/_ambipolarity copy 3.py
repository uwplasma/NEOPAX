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
    er_min, er_max = float(Er_range[0]), float(Er_range[1])
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
        x_final, _, _, _, _, active_final = lax.fori_loop(0, n_refine, body, carry0)
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



def plot_roots(rho, roots_3, entropies_3, best_root, output_dir: Path) -> Path:
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

    # --- Optionally: Plot Er profile from NTSS_Initial_Er_Opt.h5 if available ---
    input_path = os.path.join(output_dir, '../inputs/')
    er_file = os.path.join(input_path, 'NTSS_Initial_Er_Opt.h5')
    try:
        import h5py
        import interpax
        if os.path.isfile(er_file):
            with h5py.File(er_file, 'r') as f:
                r_data = f['r'][()]
                er_data = f['Er'][()]
                if len(er_data) != len(rho):
                    er_interp = interpax.interp1d(r_data, er_data, rho)
                else:
                    er_interp = er_data
                ax1.plot(rho, er_interp, color='tab:red', linewidth=2.0, linestyle=':', label='Er from NTSS_Initial_Er_Opt.h5')
    except Exception as e:
        print(f"Could not plot Er from NTSS_Initial_Er_Opt.h5: {e}")

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

    root_finder = AMBIPOLARITY_MODEL_REGISTRY.get(model_name)
    if root_finder is None:
        raise ValueError(f"Unknown ambipolarity model: {model_name}")

    def gamma_func_factory(i):
        def gamma(er):
            er_vec = Er.at[i].set(er) if n_radial > 1 else jnp.array([er])
            fluxes = flux_model(dataclasses.replace(state, Er=er_vec), params=params)
            q = params["species"].charge_qp
            Gamma = fluxes.get("Gamma_total") or fluxes.get("Gamma")
            if Gamma is None:
                raise ValueError("Flux model did not return 'Gamma' or 'Gamma_total'.")
            return jnp.sum(q[:, None] * Gamma)
        return gamma

    def entropy_func_factory(i):
        def entropy(er):
            er_vec = Er.at[i].set(er) if n_radial > 1 else jnp.array([er])
            fluxes = flux_model(dataclasses.replace(state, Er=er_vec), params=params)
            Gamma = fluxes.get("Gamma_total") or fluxes.get("Gamma")
            if Gamma is None:
                raise ValueError("Flux model did not return 'Gamma' or 'Gamma_total'.")
            return jnp.sum(jnp.abs(Gamma))
        return entropy

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

    # Blocked or pure vmap, JAX-friendly
    if er_ambipolar_blocksize is None or er_ambipolar_blocksize >= n_radial:
        # Pure vmap
        result = jax.vmap(root_finder_for_radius)(jnp.arange(n_radial))
    else:
        # Blocked vmap for memory efficiency, JAX-friendly using lax.fori_loop with static indices
        block_size = er_ambipolar_blocksize
        n_blocks = (n_radial + block_size - 1) // block_size
        # Get output shapes by running one block
        idxs0 = jnp.arange(block_size)
        roots0, entropies0, best0, n_roots0 = jax.vmap(root_finder_for_radius)(idxs0)
        roots_shape = (n_radial,) + roots0.shape[1:]
        entropies_shape = (n_radial,) + entropies0.shape[1:]
        best_shape = (n_radial,) if best0.shape == () else (n_radial,) + best0.shape[1:]
        n_roots_shape = (n_radial,) + n_roots0.shape[1:]

        def init_arrays():
            return (
                jnp.zeros(roots_shape, roots0.dtype),
                jnp.zeros(entropies_shape, entropies0.dtype),
                jnp.zeros(best_shape, best0.dtype),
                jnp.zeros(n_roots_shape, dtype=jnp.int32),
            )

        def body_fun(b, carry):
            roots_all, entropies_all, best_roots, n_roots_all = carry
            start = b * block_size
            n_valid = jnp.minimum(block_size, n_radial - start)
            idxs = start + jnp.arange(block_size)
            roots_b, entropies_b, best_b, n_roots_b = jax.vmap(root_finder_for_radius)(idxs)
            # Mask out invalid entries (beyond n_radial) by zeroing or NaN
            def mask_block(arr):
                mask = jnp.arange(block_size) < n_valid
                if arr.ndim == 1:
                    return jnp.where(mask, arr, jnp.nan)
                else:
                    # For 2D or higher, mask first axis
                    mask_shape = (block_size,) + (1,) * (arr.ndim - 1)
                    return jnp.where(mask.reshape(mask_shape), arr, jnp.nan)
            roots_b = mask_block(roots_b)
            entropies_b = mask_block(entropies_b)
            best_b = mask_block(best_b)
            n_roots_b = mask_block(n_roots_b)
            # Defensive: ensure n_roots_b is int32 for JAX dynamic_update_slice
            n_roots_b = n_roots_b.astype(jnp.int32)
            # Use dynamic_update_slice to update arrays
            roots_all = lax.dynamic_update_slice(roots_all, roots_b, (start,) + (0,) * (roots_all.ndim - 1))
            entropies_all = lax.dynamic_update_slice(entropies_all, entropies_b, (start,) + (0,) * (entropies_all.ndim - 1))
            best_roots = lax.dynamic_update_slice(best_roots, best_b, (start,) + (0,) * (best_roots.ndim - 1))
            n_roots_all = lax.dynamic_update_slice(n_roots_all, n_roots_b, (start,) + (0,) * (n_roots_all.ndim - 1))
            return (roots_all, entropies_all, best_roots, n_roots_all)

        roots_all, entropies_all, best_roots, n_roots_all = lax.fori_loop(
            0, n_blocks, body_fun, init_arrays()
        )
        result = (roots_all, entropies_all, best_roots, n_roots_all)
    return result

# --- Orchestration/config reading only ---
def solve_ambipolarity_roots_from_config(state, config, params):
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
    # Select transport/entropy models
    flux_model_name = neoclassical_cfg.get("flux_model", "monkes_database")
    entropy_model_name = neoclassical_cfg.get("entropy_model", flux_model_name)
    flux_model = build_transport_flux_model(flux_model_name)
    entropy_model = get_entropy_model(entropy_model_name)

    # Call the JIT/diff radial root-finder
    roots_all, entropies_all, best_roots, n_roots_all = solve_ambipolarity_roots_radial(
        state, config, params, model_name, flux_model, entropy_model, amb_cfg
    )

    # Read plotting/output options from config
    er_ambipolar_plot = amb_cfg.get("er_ambipolar_plot", False)
    er_ambipolar_write_hdf5 = amb_cfg.get("er_ambipolar_write_hdf5", False)
    er_ambipolar_output_dir = amb_cfg.get("er_ambipolar_output_dir", None)

    n_radial = n_roots_all.shape[0] if hasattr(n_roots_all, "shape") else 1
    n_roots_all = jnp.atleast_1d(n_roots_all)
    if n_roots_all.shape[0] != n_radial:
        n_roots_all = jnp.broadcast_to(n_roots_all, (n_radial,))
    return roots_all, entropies_all, best_roots, n_roots_all, er_ambipolar_plot, er_ambipolar_write_hdf5, er_ambipolar_output_dir


# Register the numerical methods as ambipolarity models

register_ambipolarity_model("two_stage", find_ambipolar_Er_min_entropy_jit_multires)

register_ambipolarity_model("adaptive", find_ambipolar_Er_min_entropy_jit_adaptive)
