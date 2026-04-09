# Two-Stage Adaptive Ambipolar Root-Finding Refactor

## Motivation
- Reduce memory usage and speed up root-finding by only refining where roots are bracketed.
- Use a coarse scan (n_coarse) to bracket roots, then refine each bracket with Newton-bisection (n_refine).
- Pad to a fixed number of roots (max_roots, default 3) for JIT/static shape compatibility.
- Avoid storing large arrays for all scan points; only keep roots and entropies.

## Implementation
-Please check all the _ambipolairty.py file for corrrect placement before doing any changes 
- Add´'find_ambipolar_Er_min_entropy_jit_multires` for a single radius (coarse/fine, padded, JIT-compatible).
- Add `find_all_ambipolar_Er_roots_profile_jit` as the default vectorized profile root-finder, using the two-stage method.
- Removed legacy dense scan from main workflow (still available as `find_ambipolar_Er_min_entropy_jit` for explicit use).
- All code is JIT and autodiff compatible (JAX vmap/lax.scan, no Python loops).
- All routines pad to `max_roots` for static shape.

## Usage
- Use `find_all_ambipolar_Er_roots_profile_jit` for all profile root-finding (now two-stage by default).
- Set `n_coarse`, `n_refine`, and `max_roots` as needed (defaults: 24, 8, 3).
- Only roots and entropies at root locations are stored.
-Please check that new inputs are added correctly to toml reading and plot example



## Benefits
- Memory usage is now O(n_radial * max_roots) instead of O(n_radial * n_scan).
- Much faster for difficult cases with few roots.
- Fully differentiable and JIT-compatible.

## Example
```python
roots_all, entropies_all, best_roots, n_roots_all = find_all_ambipolar_Er_roots_profile_jit(
    get_Neoclassical_Fluxes, species, grid, field, database, state,
    er_min, er_max, n_coarse=24, n_refine=8, max_roots=3)
```

## Migration Notes
- If you were using `n_scan`, switch to `n_coarse` and `n_refine`.
- If you need the legacy dense scan, call `find_ambipolar_Er_min_entropy_jit` directly.
- Update plotting/diagnostic scripts to use the new interface.
