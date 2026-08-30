"""Legacy constructed-QI/max-J compatibility implementation.

This is the NEOPAX-owned copy of the direct Boozer-data helper formerly in
``vmex.core.omnigenity_j``.  It deliberately contains no VMEC or Boozer call:
callers supply the one Boozer spectrum already produced by the NEOPAX path.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import jax
import jax.numpy as jnp

Array = Any


def _as_1d(values):
    return jnp.atleast_1d(jnp.asarray(values, dtype=jnp.float64))


def _soft_min_idx(values, beta: float = 50.0):
    values = jnp.asarray(values, dtype=jnp.float64)
    weights = jax.nn.softmax(-jnp.asarray(beta, dtype=values.dtype) * values)
    return jnp.sum(jnp.arange(values.shape[0], dtype=values.dtype) * weights)


def _cummin(values):
    return jax.lax.associative_scan(jnp.minimum, jnp.asarray(values, dtype=jnp.float64))


def _smooth_signed_sqrt(values, eps: float = 1.0e-9):
    values = jnp.asarray(values, dtype=jnp.float64)
    eps_arr = jnp.asarray(eps, dtype=values.dtype)
    abs_smooth = jnp.sqrt(values * values + eps_arr * eps_arr)
    return values / jnp.sqrt(abs_smooth + eps_arr)


def _smooth_positive_sqrt(values, eps: float = 1.0e-10):
    values = jnp.asarray(values, dtype=jnp.float64)
    eps_arr = jnp.asarray(eps, dtype=values.dtype)
    return jnp.sqrt(jnp.maximum(values, 0.0) + eps_arr) - jnp.sqrt(eps_arr)


def _smooth_abs_power(values, exponent: float, eps: float = 1.0e-12):
    values = jnp.asarray(values, dtype=jnp.float64)
    eps_arr = jnp.asarray(eps, dtype=values.dtype)
    return jnp.power(values * values + eps_arr * eps_arr, 0.5 * float(exponent))


def _periodic_angle_distance(alpha_a, alpha_b):
    two_pi = jnp.asarray(2.0 * jnp.pi, dtype=jnp.result_type(alpha_a, alpha_b))
    delta = jnp.abs(alpha_a - alpha_b)
    return jnp.minimum(delta, two_pi - delta)


def _apply_smooth_goodman_transform(b_line, phi_coords):
    b_line = jnp.asarray(b_line, dtype=jnp.float64)
    phi_coords = jnp.asarray(phi_coords, dtype=jnp.float64)
    n = int(b_line.shape[0])
    indices = jnp.arange(n, dtype=b_line.dtype)
    s_indmin = _soft_min_idx(b_line)
    split_beta, peak_beta, cap_beta = (jnp.asarray(v, dtype=b_line.dtype) for v in (10.0, 120.0, 35.0))
    mask_l = jax.nn.sigmoid(split_beta * (s_indmin - indices))
    mask_r = 1.0 - mask_l
    lhs_gate = jax.nn.sigmoid(split_beta * (s_indmin - indices))
    lhs_weights = jax.nn.softmax(peak_beta * (b_line - 10.0 * (1.0 - lhs_gate)))
    lhs_peak_val, lhs_peak_idx = jnp.sum(lhs_weights * b_line), jnp.sum(lhs_weights * indices)
    bl_base = jax.nn.sigmoid(cap_beta * (lhs_peak_idx - indices)) * lhs_peak_val + (1.0 - jax.nn.sigmoid(cap_beta * (lhs_peak_idx - indices))) * b_line
    bl_sq = _cummin(bl_base)
    rhs_gate = jax.nn.sigmoid(split_beta * (indices - s_indmin))
    rhs_weights = jax.nn.softmax(peak_beta * (b_line - 10.0 * (1.0 - rhs_gate)))
    rhs_peak_val, rhs_peak_idx = jnp.sum(rhs_weights * b_line), jnp.sum(rhs_weights * indices)
    after_peak = jax.nn.sigmoid(cap_beta * (indices - rhs_peak_idx))
    br_sq = jnp.flip(_cummin(jnp.flip(after_peak * rhs_peak_val + (1.0 - after_peak) * b_line)))
    b_min_val = jnp.interp(s_indmin, indices, b_line)
    phi_mid, phi_start, phi_end = jnp.interp(s_indmin, indices, phi_coords), phi_coords[0], phi_coords[-1]
    x1_l = (phi_coords - phi_start) / (phi_mid - phi_start + 1.0e-10)
    x1_r = (phi_coords - phi_mid) / (phi_end - phi_mid + 1.0e-10)
    shape_l, shape_r = (jnp.cos(2.0 * jnp.pi * x1_l) + 1.0) / 2.0, (jnp.cos(2.0 * jnp.pi * x1_r) + 1.0) / 2.0
    f_l = jnp.where(x1_l < 0.5, (1.0 - bl_sq) * shape_l**50.0, -b_min_val * shape_l**15.0)
    f_r = jnp.where(x1_r < 0.5, -b_min_val * shape_r**15.0, (1.0 - rhs_peak_val) * shape_r**50.0)
    out = jnp.clip(mask_l * (bl_sq + f_l) + mask_r * (br_sq + f_r), 0.0, 1.0)
    x_edge = (phi_coords - phi_start) / (phi_end - phi_start + 1.0e-10)
    left_edge = jax.nn.sigmoid(120.0 * (0.015 - x_edge))
    right_edge = jax.nn.sigmoid(120.0 * (x_edge - 0.985))
    edge_blend = left_edge + right_edge - left_edge * right_edge
    return (1.0 - edge_blend) * out + edge_blend


def _branch_crossings(phi_coords, b_line, bj_level):
    phi_coords, b_line, bj_level = (jnp.asarray(value, dtype=jnp.float64) for value in (phi_coords, b_line, bj_level))
    indices = jnp.arange(b_line.shape[0], dtype=jnp.float64)
    s_indmin = _soft_min_idx(b_line)
    left_mask = jax.nn.sigmoid(10.0 * (s_indmin - indices))

    def _invert_branch(phi_branch, b_branch, branch_mask):
        phi0, phi1, b0, b1 = phi_branch[:-1], phi_branch[1:], b_branch[:-1], b_branch[1:]
        db = b1 - b0
        scale = jnp.maximum(5.0e-3 * jnp.max(jnp.abs(b_branch)), 1.0e-6)
        valid = branch_mask[:-1] * branch_mask[1:] * jax.nn.sigmoid(60.0 * db / scale)
        lo, hi = jnp.minimum(b0, b1), jnp.maximum(b0, b1)
        valid = valid * jax.nn.sigmoid(40.0 * (bj_level - lo) / scale) * jax.nn.sigmoid(40.0 * (hi - bj_level) / scale) * jnp.tanh(jnp.abs(db) / scale) ** 2 + 1.0e-14
        t = jnp.clip((bj_level - b0) / (db + 1.0e-12), 0.0, 1.0)
        return jnp.sum(valid * (phi0 + t * (phi1 - phi0))) / jnp.sum(valid)

    phi_min = jnp.interp(s_indmin, indices, phi_coords)
    phi_lo = _invert_branch(jnp.flip(phi_coords), jnp.flip(b_line), jnp.flip(left_mask))
    phi_hi = _invert_branch(phi_coords, b_line, 1.0 - left_mask)
    low_blend = jax.nn.sigmoid(250.0 * (0.01 - bj_level))
    high_blend = jax.nn.sigmoid(250.0 * (bj_level - 0.99))
    phi_lo, phi_hi = (1.0 - low_blend) * phi_lo + low_blend * phi_min, (1.0 - low_blend) * phi_hi + low_blend * phi_min
    return (1.0 - high_blend) * phi_lo + high_blend * phi_coords[0], (1.0 - high_blend) * phi_hi + high_blend * phi_coords[-1]


def _compute_j_pair(phi_coords, b_input, b_target, bj_levels, gi_value, *, nphi_int: int = 128):
    b_input, b_target, phi_coords, bj_levels, gi_value = (jnp.asarray(value, dtype=jnp.float64) for value in (b_input, b_target, phi_coords, bj_levels, gi_value))
    bmin, bmax = jnp.min(b_target), jnp.max(b_target)
    scale = jnp.maximum(bmax - bmin, 1.0e-12)
    target_norm, bj_norm = (b_target - bmin) / scale, (bj_levels - bmin) / scale
    p1, p2 = jax.vmap(lambda bj: _branch_crossings(phi_coords, target_norm, bj))(bj_norm)
    t = jnp.linspace(0.0, 1.0, int(nphi_int), dtype=b_target.dtype)
    phi_grid = p1[:, None] + t[None, :] * (p2 - p1)[:, None]
    bi_g, bc_g = jnp.interp(phi_grid, phi_coords, b_input), jnp.interp(phi_grid, phi_coords, b_target)
    bj_v, metric_factor = jnp.maximum(bj_levels[:, None], 1.0e-9), gi_value / jnp.maximum(bi_g, 1.0e-9)
    ji = jnp.trapezoid(_smooth_signed_sqrt(1.0 - bi_g / bj_v) * metric_factor, x=phi_grid, axis=1)
    jc = jnp.trapezoid(_smooth_positive_sqrt(1.0 - bc_g / bj_v) * metric_factor, x=phi_grid, axis=1)
    return ji, jc


def _synthesize_boozer_field_lines(*, bmnc_b, xm_b, xn_b, iota_b, nfp: int, nphi: int, nalpha: int):
    bmnc_b = jnp.asarray(bmnc_b, dtype=jnp.float64)
    xm_b, xn_b = jnp.asarray(np.asarray(xm_b, dtype=float)), jnp.asarray(np.asarray(xn_b, dtype=float))
    iota_b = jnp.atleast_1d(jnp.asarray(iota_b, dtype=jnp.float64))
    phi = jnp.linspace(0.0, 2.0 * np.pi / float(nfp), int(nphi), endpoint=True, dtype=bmnc_b.dtype)
    alpha = jnp.linspace(0.0, 2.0 * jnp.pi, int(nalpha), endpoint=False, dtype=bmnc_b.dtype)
    theta = alpha[None, :, None] + iota_b[:, None, None] * phi[None, None, :]
    return phi, alpha, jnp.einsum("sapm,sm->sap", jnp.cos(theta[..., None] * xm_b - phi[None, None, :, None] * xn_b), bmnc_b)


def j_invariant_qi_maxj_residual_from_boozer(*, bmnc_b, xm_b, xn_b, iota_b, gi_b, s_b, nfp: int, weights: Iterable[float] | None = None, nphi: int = 101, nalpha: int = 51, n_bounce: int = 66, p_j: float = 1.0, p_lambda: float = 1.0, nphi_int: int = 128, target_maxj: float = -0.06, qi_weight: float = 1.0, maxj_weight: float = 1.0, include_qi: bool = True, include_maxj: bool = True, maxj_pairing: str = "same_alpha", maxj_sigma_alpha: float | None = None) -> dict[str, Array]:
    """Return legacy constructed-QI/max-J residual blocks from one Boozer spectrum."""
    bmnc_b, iota_b, gi_b, s_b = (jnp.asarray(value, dtype=jnp.float64) for value in (bmnc_b, iota_b, gi_b, s_b))
    nsurf = int(bmnc_b.shape[0])
    if nsurf == 0 or int(nphi) < 8 or int(nalpha) < 2 or int(n_bounce) < 2:
        raise ValueError("legacy QI/max-J requires non-empty surfaces, nphi >= 8, nalpha >= 2, n_bounce >= 2")
    w_arr = jnp.ones((nsurf,), dtype=jnp.float64) if weights is None else _as_1d(weights)
    if int(w_arr.shape[0]) != nsurf:
        raise ValueError("weights must have the same length as the Boozer surfaces")
    if maxj_pairing not in {"all_to_all", "same_alpha", "soft_local"}:
        raise ValueError("invalid maxj_pairing")
    phi, alpha, b_lines = _synthesize_boozer_field_lines(bmnc_b=bmnc_b, xm_b=xm_b, xn_b=xn_b, iota_b=iota_b, nfp=nfp, nphi=nphi, nalpha=nalpha)
    bj_norm = jnp.power(jnp.arange(int(n_bounce), dtype=jnp.float64) / jnp.maximum(int(n_bounce) - 1, 1), float(p_lambda))
    sigma_alpha = max(float(maxj_sigma_alpha) if maxj_sigma_alpha is not None else 2.0 * np.pi / max(int(nalpha), 1), 1.0e-8)
    def _per_line(b_line, gi_surface):
        bmin, bmax = jnp.min(b_line), jnp.max(b_line)
        scale = jnp.maximum(bmax - bmin, 1.0e-10)
        target = _apply_smooth_goodman_transform((b_line - bmin) / scale, phi) * scale + bmin
        return _compute_j_pair(phi, b_line, target, bj_norm * scale + bmin, gi_surface, nphi_int=nphi_int)
    ji_all, jc_all = jax.vmap(lambda bs, gs: jax.vmap(lambda line: _per_line(line, gs))(bs))(b_lines, gi_b)
    diagnostics: dict[str, Array] = {"phi": phi, "alpha": alpha, "surfaces": s_b, "ji": ji_all, "jc": jc_all}
    blocks = []
    if include_qi:
        ji_pow, jc_pow = ji_all ** float(p_j), jc_all ** float(p_j)
        sum_ji, sum_jc = jnp.sum(ji_pow, axis=1), jnp.sum(jc_pow, axis=1)
        qi_num = jnp.sum(float(nalpha) * (jnp.sum(ji_pow * ji_pow, axis=1) + jnp.sum(jc_pow * jc_pow, axis=1)) - 2.0 * sum_ji * sum_jc, axis=1)
        qi_surface = jnp.sqrt(jnp.maximum(qi_num, 0.0) / ((jnp.sum(sum_ji + sum_jc, axis=1) / (2.0 * float(n_bounce))) ** 2 + 1.0e-10))
        qi_block = float(qi_weight) * jnp.sqrt(w_arr) * qi_surface
        blocks.append(qi_block); diagnostics.update(qi_surface=qi_surface, qi_objective=jnp.sum(qi_block * qi_block), ji_pow=ji_pow, jc_pow=jc_pow)
    if include_maxj:
        if nsurf < 2:
            slope, maxj_surface, maxj_pair_weights = jnp.zeros((0, 0, 0)), jnp.zeros((0,)), jnp.zeros((0, 0)); jc_pow_maxj = jnp.zeros_like(jc_all); maxj_block = jnp.zeros((0,))
        else:
            jc_pow_maxj = _smooth_abs_power(jc_all, float(p_j)); ds = jnp.where(jnp.abs(s_b[1:] - s_b[:-1]) > 0.0, s_b[1:] - s_b[:-1], 1.0e-10)
            if maxj_pairing == "all_to_all": alpha_weights = jnp.ones((nalpha, nalpha), dtype=jnp.float64) / float(nalpha)
            elif maxj_pairing == "same_alpha": alpha_weights = jnp.eye(nalpha, dtype=jnp.float64)
            else: alpha_weights = jax.nn.softmax(-_periodic_angle_distance(alpha[:, None], alpha[None, :]) ** 2 / (2.0 * sigma_alpha * sigma_alpha), axis=1)
            def _pair(hi, lo, ds_i):
                if maxj_pairing == "soft_local":
                    pair_slope = (hi[:, :, None] - jnp.swapaxes(lo, 0, 1)[None, :, :]) / (ds_i * (0.5 * (hi[:, :, None] + jnp.swapaxes(lo, 0, 1)[None, :, :]) + 1.0e-10))
                    return jnp.einsum("abl,al->ab", pair_slope, alpha_weights), jnp.sqrt(jnp.einsum("abl,al->ab", jnp.maximum(0.0, pair_slope - float(target_maxj)) ** 2, alpha_weights))
                matched = alpha_weights @ lo; local = (hi - matched) / (ds_i * (0.5 * (hi + matched) + 1.0e-10)); return local, jnp.maximum(0.0, local - float(target_maxj))
            slope, violation = jax.vmap(_pair)(jc_pow_maxj[1:, :, 1:], jc_pow_maxj[:-1, :, 1:], ds)
            maxj_surface = jnp.sqrt(jnp.sum(violation ** 2, axis=(1, 2))); maxj_block = float(maxj_weight) * jnp.sqrt(0.5 * (w_arr[:-1] + w_arr[1:])) * maxj_surface; maxj_pair_weights = alpha_weights
        blocks.append(maxj_block); diagnostics.update(maxj_surface=maxj_surface, maxj_objective=jnp.sum(maxj_block * maxj_block), jc_pow_maxj=jc_pow_maxj, maxj_slope=slope, maxj_pair_weights=maxj_pair_weights)
    if not blocks: raise ValueError("At least one of include_qi/include_maxj must be True.")
    residuals1d = jnp.concatenate(blocks)
    return {"residuals1d": residuals1d, "total": jnp.sum(residuals1d * residuals1d), "residual_block_sizes": jnp.asarray([block.size for block in blocks], dtype=jnp.int32), **diagnostics}
