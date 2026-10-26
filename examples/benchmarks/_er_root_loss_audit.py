"""Candidate and original root/bootstrap costs for the full-transport AD benchmark.

The caller supplies the real terminal state and the established reverse
kernel differentiates these rows. Synthetic-array tests do not validate
their physical parameter sensitivities. Audit cost rows are costs, not
least-squares residuals for use in an optimizer.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from NEOPAX._reverse_ad_parameters import (
    parse_vmec_boundary_parameter_spec, vmex_boundary_parameter_scales,
)
from examples.benchmarks import compare_er_transition_loss_candidates as candidates


AUDIT_OBJECTIVES = (
    "audit_root_interpolated_Er",
    "audit_root_local_cost",
    "audit_root_annular_cost",
    "audit_root_nearest_downcrossing_rho",
)
CANDIDATE_OBJECTIVES = AUDIT_OBJECTIVES[1:3]
ROOT_CREATION_OBJECTIVES = (*AUDIT_OBJECTIVES, "audit_root_inner_mean_Er")
ORIGINAL_RAW_OBJECTIVES = ("Er_transition_location_moment", "Er_transition_strength",
                           "bootstrap_current_softmax_abs_scaled")
ORIGINAL_CONTROL_OBJECTIVES = (*ROOT_CREATION_OBJECTIVES, *ORIGINAL_RAW_OBJECTIVES)
ORIGINAL_COST_OBJECTIVES = ("audit_root_original_cost", "audit_root_bootstrap_cost",
                            "audit_root_original_bootstrap_cost")


def original_objective_settings():
    example = candidates.audit.example
    return dict(position_weight=float(example.ER_TRANSITION_RHO_WEIGHT),
                strength_weight=float(example.ER_TRANSITION_STRENGTH_WEIGHT),
                strength_target=float(example.ER_TRANSITION_STRENGTH_TARGET),
                bootstrap_weight=float(example.BOOTSTRAP_WEIGHT),
                bootstrap_limit=float(example.BOOTSTRAP_LIMIT_SCALED),
                temperature_kv_m=float(example.ER_TRANSITION_TEMPERATURE_KV_M),
                rho_softness=float(example.ER_TRANSITION_RHO_SOFTNESS),
                softmax_beta=float(example.ER_TRANSITION_SOFTMAX_BETA))


def original_transition_values(er, rho, settings):
    """Use the production selector, including its target-centered prior."""
    from NEOPAX._reverse_ad_transport import smooth_positive_to_negative_er_transition_metrics
    options = original_objective_settings()
    location, strength = smooth_positive_to_negative_er_transition_metrics(
        er, rho, rho_min=settings.inner_radius, rho_max=settings.outer_radius,
        rho_prior=settings.target, temperature_kv_m=options["temperature_kv_m"],
        rho_softness=options["rho_softness"], softmax_beta=options["softmax_beta"])
    return jnp.stack((location-settings.target, strength))


def original_cost_components(raw_values):
    """Original weighted least-squares costs, never squared a second time."""
    moment, strength, bootstrap = jnp.asarray(raw_values)
    options = original_objective_settings()
    deficit = candidates.audit.example.smooth_positive_part(options["strength_target"]-strength)
    excess = jnp.maximum(bootstrap-options["bootstrap_limit"], 0.)
    return jnp.stack((.5*options["position_weight"]*moment**2,
                      .5*options["strength_weight"]*deficit**2,
                      .5*options["bootstrap_weight"]*excess**2))


def original_cost_vector(raw_values):
    position, strength, bootstrap = original_cost_components(raw_values)
    return jnp.stack((position+strength, bootstrap, position+strength+bootstrap))


def add_original_metrics(metrics, raw_values):
    """Attach original observable values and separate component costs."""
    raw = np.asarray(raw_values, dtype=float)
    position, strength, bootstrap = map(float, original_cost_components(raw))
    options = original_objective_settings()
    metrics["original_objectives"] = dict(
        Er_transition_rho=float(raw[0]+metrics["target_rho"]),
        Er_transition_location_moment=float(raw[0]), Er_transition_strength=float(raw[1]),
        strength_deficit=float(candidates.audit.example.smooth_positive_part(options["strength_target"]-raw[1])),
        bootstrap_current_softmax_abs_scaled=float(raw[2]),
        bootstrap_penalty=float(max(raw[2]-options["bootstrap_limit"], 0.)),
        **options)
    metrics["candidates"].update({
        ORIGINAL_COST_OBJECTIVES[0]: dict(cost=position+strength, position_cost=position, strength_cost=strength),
        ORIGINAL_COST_OBJECTIVES[1]: dict(cost=bootstrap, bootstrap_cost=bootstrap),
        ORIGINAL_COST_OBJECTIVES[2]: dict(cost=position+strength+bootstrap, position_cost=position,
                                         strength_cost=strength, bootstrap_cost=bootstrap)})
    return metrics


def experiment_inputs(config, values, initial_er_root_ad):
    """Save concrete inputs so forward descent checks need no second AD run."""
    def encode(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (np.ndarray, np.generic)):
            return value.tolist()
        raise TypeError(f"Non-concrete configuration value: {type(value).__name__}")
    return dict(resolved_config=json.loads(json.dumps(config, default=encode)),
                parameter_values=np.asarray(values, dtype=float).tolist(), initial_er_root_ad=initial_er_root_ad)


@dataclass(frozen=True)
class RootLossAuditSettings:
    target: float = 0.514
    inner_radius: float = 0.30
    outer_radius: float = 0.9
    profile_trust_multiplier: float = 3.7
    geometry_ess_alpha: float = 1.2

    def __post_init__(self):
        if not all(math.isfinite(value) for value in (self.target, self.inner_radius, self.outer_radius)):
            raise ValueError("Root audit radii must be finite.")
        if not 0 <= self.inner_radius < self.target < self.outer_radius <= 1:
            raise ValueError("Root audit requires 0 <= inner < target < outer <= 1.")
        if not math.isfinite(self.profile_trust_multiplier) or self.profile_trust_multiplier <= 0:
            raise ValueError("Profile trust multiplier must be finite and positive.")
        if not math.isfinite(self.geometry_ess_alpha) or self.geometry_ess_alpha < 0:
            raise ValueError("Geometry ESS alpha must be finite and nonnegative.")


def candidate_residuals(er, rho, settings):
    """Two new residual blocks, each with one overall weight of one."""
    zero = jnp.atleast_1d(jnp.interp(settings.target, rho, er) / candidates.ER_SCALE)
    left = jnp.clip(jnp.searchsorted(rho, settings.target, side="left") - 1, 0, rho.size - 2)
    right = jnp.clip(jnp.searchsorted(rho, settings.target, side="right"), 1, rho.size - 1)
    signed = jnp.stack((er[left], -er[right])) / candidates.ER_SCALE
    deficits = candidates.audit.example.smooth_positive_part(candidates.MARGIN - signed)
    return {
        "audit_root_local_cost": jnp.concatenate((zero, deficits / jnp.sqrt(2.0))),
        "audit_root_annular_cost": candidates.target_zero_region_residuals(
            er, rho, rho_min=settings.inner_radius, rho_max=settings.outer_radius, rho_target=settings.target,
        ),
    }


def parameter_scales(parameter_labels, profile_cfg, settings):
    """Physical deltas per unit trust coordinate, matching the simple optimizer.

    Nominal profile scales include its block trust multiplier; VMEC harmonics
    use the shared ESS scaling implementation. This is a diagonal mapping.
    """
    defaults = dict(n0=4.21, T0=17.8, density_shape_power=2.0, temperature_shape_power=2.0,
                    density_shape_alpha=1.0, temperature_shape_alpha=1.0)
    scales = []
    for name in parameter_labels:
        if ":" in name:
            spec = parse_vmec_boundary_parameter_spec(name)
            scales.append(vmex_boundary_parameter_scales((spec,), ess_alpha=settings.geometry_ess_alpha)[0])
        else:
            raw = profile_cfg.get(name, defaults[name])
            if isinstance(raw, (list, tuple)):
                raw = raw[0]
            scales.append(max(abs(float(raw)), 1e-12) * settings.profile_trust_multiplier)
    return np.asarray(scales)


def terminal_metrics(er, rho, settings):
    """Concrete root, sign and amplitude observations; no flux evaluations."""
    er, rho = np.asarray(er, dtype=float), np.asarray(rho, dtype=float)
    if er.ndim != 1 or er.shape != rho.shape or len(rho) < 3:
        raise ValueError("Expected matching one-dimensional Er and rho arrays with at least three nodes.")
    if not np.isfinite(er).all() or not np.isfinite(rho).all() or not np.all(np.diff(rho) > 0):
        raise ValueError("Expected finite Er and a finite, strictly increasing rho grid.")
    if not rho[0] < settings.target < rho[-1]:
        raise ValueError("The target must have grid nodes on both sides.")
    mask = (rho >= settings.inner_radius) & (rho <= settings.outer_radius)
    if not np.any(mask & (rho < settings.target)) or not np.any(mask & (rho > settings.target)):
        raise ValueError("The annulus must resolve both sides of the target.")
    # Include crossings by their interpolated radius, rather than the face
    # centre, so observation and differentiation use identical boundaries.
    brackets = np.flatnonzero((er[:-1] >= 0) & (er[1:] < 0))
    roots = rho[brackets] + np.diff(rho)[brackets] * er[brackets] / (er[brackets] - er[brackets + 1])
    selected = (roots >= settings.inner_radius) & (roots <= settings.outer_radius)
    roots, brackets = roots[selected], brackets[selected]
    nearest = int(np.argmin(abs(roots - settings.target))) if len(roots) else None
    blocks = candidate_residuals(jnp.asarray(er), jnp.asarray(rho), settings)
    costs = {}
    for name, block in blocks.items():
        block = np.asarray(block)
        costs[name] = dict(cost=float(0.5 * np.dot(block, block)),
                           zero_cost=float(0.5 * block[0]**2), sign_cost=float(0.5 * np.dot(block[1:], block[1:])))
    root = float(roots[nearest]) if nearest is not None else None
    inner = mask & (rho < settings.target)
    outer = mask & (rho > settings.target)
    margin = candidates.MARGIN * candidates.ER_SCALE
    return dict(Er_at_target=float(np.interp(settings.target, rho, er)),
                annular_max_Er=float(np.max(er[mask])), annular_min_Er=float(np.min(er[mask])),
                inner_mean_Er=float(np.mean(er[inner])),
                inner_positive_fraction=float(np.mean(er[inner] > 0)),
                inner_positive_margin_fraction=float(np.mean(er[inner] >= margin)),
                outer_negative_fraction=float(np.mean(er[outer] < 0)),
                outer_negative_margin_fraction=float(np.mean(er[outer] <= -margin)),
                annular_Er_rms=float(np.sqrt(np.mean(er[mask]**2))),
                wrong_sign_fraction=float(np.mean(((rho[mask] < settings.target) & (er[mask] < 0))
                                                 | ((rho[mask] > settings.target) & (er[mask] > 0)))),
                actual_downcrossings=roots.tolist(), nearest_downcrossing=root,
                nearest_downcrossing_bracket=int(brackets[nearest]) if nearest is not None else None,
                position_cost=0.5 * (root - settings.target)**2 if root is not None else None,
                candidates=costs)


def nearest_downcrossing(er, rho, settings):
    """Piecewise chord diagnostic; never average distinct crossings.

    Selection is discrete. Its derivative is conditional on the selected
    bracket. A missing crossing returns the target and zero derivative; the
    report explicitly marks it unavailable for directional interpretation.
    This row is a diagnostic, not a proposed optimization objective.
    """
    valid = (er[:-1] >= 0) & (er[1:] < 0)
    denominator = jnp.where(valid, er[:-1] - er[1:], 1.0)
    roots = rho[:-1] + jnp.diff(rho) * er[:-1] / denominator
    valid = valid & (roots >= settings.inner_radius) & (roots <= settings.outer_radius)
    index = jnp.argmin(jnp.where(valid, jnp.abs(roots - settings.target), jnp.inf))
    return jnp.where(jnp.any(valid), roots[index], jnp.asarray(settings.target, dtype=er.dtype))


def make_audit_callbacks(labels, settings):
    """Return a stable objective callback, a primal observer and one snapshot."""
    labels = tuple(labels)
    if not labels or not set(labels).issubset(ORIGINAL_CONTROL_OBJECTIVES):
        raise ValueError("Root-loss audit evaluates only new candidates, original root/bootstrap controls and their diagnostics.")
    snapshot = {}

    def observe(final_state, runtime, final_time):
        # Called once outside terminal AD by the reverse kernel. Retain only
        # copied host arrays, never JAX traces, pullbacks, or state histories.
        er = np.array(jax.device_get(final_state.Er), dtype=float, copy=True)
        rho = np.array(jax.device_get(runtime.geometry.rho_grid), dtype=float, copy=True)
        terminal_metrics(er, rho, settings)
        snapshot.clear()
        snapshot.update(Er=er, rho=rho,
                        final_time=float(jax.device_get(final_time)))

    def evaluate(final_state, runtime, objective_index):
        name = labels[int(objective_index)]
        er, rho = jnp.asarray(final_state.Er), jnp.asarray(runtime.geometry.rho_grid)
        if name == "audit_root_interpolated_Er":
            return jnp.interp(settings.target, rho, er)
        if name == "audit_root_inner_mean_Er":
            mask = (rho >= settings.inner_radius) & (rho < settings.target)
            return jnp.sum(jnp.where(mask, er, 0.0)) / jnp.sum(mask)
        if name in CANDIDATE_OBJECTIVES:
            block = candidate_residuals(er, rho, settings)[name]
            return 0.5 * jnp.vdot(block, block)
        if name == "audit_root_nearest_downcrossing_rho":
            return nearest_downcrossing(er, rho, settings)
        if name in ORIGINAL_RAW_OBJECTIVES[:2]:
            return original_transition_values(er, rho, settings)[ORIGINAL_RAW_OBJECTIVES.index(name)]
        if name == ORIGINAL_RAW_OBJECTIVES[2]:
            # Keep the native name so the reverse kernel uses its established
            # compact momentum-corrected bootstrap transpose, not an ordinary
            # custom-objective VJP through the full flux calculation.
            from NEOPAX._reverse_ad_transport import bootstrap_current_softmax_abs_scaled
            return bootstrap_current_softmax_abs_scaled(final_state, runtime)
        raise ValueError(f"Unknown root-loss audit row: {name}")

    return evaluate, observe, snapshot


def summarize_audit(*, residual_labels, parameter_labels, values, jacobian, profile_cfg, settings, snapshot):
    """Judge cost-descent directions against actual chord-root movement.

    A candidate changing only amplitudes can have a large gradient and zero
    predicted root progress. Report profile and geometry groups separately,
    without comparing raw magnitudes across incompatible parameter units.
    """
    labels = tuple(str(label).split(":", 1)[-1] for label in residual_labels)
    # transport: is stripped above; geometry labels remain harmless extras.
    parameter_labels = tuple(parameter_labels)
    values, jacobian = np.asarray(values, dtype=float), np.asarray(jacobian, dtype=float)
    row = {name: i for i, name in enumerate(labels)}
    if not snapshot:
        raise RuntimeError("Root-loss audit did not receive the physical terminal-state snapshot.")
    er, rho = snapshot["Er"], snapshot["rho"]
    metrics = terminal_metrics(er, rho, settings)
    root = metrics["nearest_downcrossing"]
    has_root = root is not None
    root_gradient = jacobian[row["audit_root_nearest_downcrossing_rho"]]
    target_field_gradient = jacobian[row["audit_root_interpolated_Er"]]
    inner_field_gradient = (jacobian[row["audit_root_inner_mean_Er"]]
                            if "audit_root_inner_mean_Er" in row else None)
    scales = parameter_scales(parameter_labels, profile_cfg, settings)

    def observable(name):
        return float(values[row[name]]), jacobian[row[name]]

    costs = {name: observable(name) for name in CANDIDATE_OBJECTIVES}
    if set(ORIGINAL_RAW_OBJECTIVES) & row.keys():
        if not set(ORIGINAL_RAW_OBJECTIVES).issubset(row):
            raise ValueError("The original controls require position, strength and native bootstrap rows.")
        raw = np.asarray([values[row[name]] for name in ORIGINAL_RAW_OBJECTIVES])
        raw_jacobian = np.asarray([jacobian[row[name]] for name in ORIGINAL_RAW_OBJECTIVES])
        scalar_costs = np.asarray(original_cost_vector(raw))
        gradients = np.asarray(jax.jacrev(original_cost_vector)(jnp.asarray(raw))) @ raw_jacobian
        costs.update({name: (value, gradient) for name, value, gradient in
                      zip(ORIGINAL_COST_OBJECTIVES, scalar_costs, gradients, strict=True)})
        metrics["target_rho"] = settings.target
        add_original_metrics(metrics, raw)
    comparisons = []
    for name, (cost, gradient) in costs.items():
        groups = {}
        for group, indices in (
            ("profiles", [i for i, p in enumerate(parameter_labels) if ":" not in p]),
            ("geometry", [i for i, p in enumerate(parameter_labels) if ":" in p]),
        ):
            g = np.asarray(gradient)[indices] * scales[indices]
            gr = root_gradient[indices] * scales[indices]
            norm = float(np.linalg.norm(g))
            rate = float(-np.dot(gr, g) / norm) if has_root and norm > 0 else None
            er_rate = float(-np.dot(target_field_gradient[indices] * scales[indices], g) / norm) if norm > 0 else None
            error_rate = (root - settings.target) * rate if rate is not None else None
            physical_direction = np.zeros(len(parameter_labels))
            if norm > 0:
                physical_direction[indices] = -scales[indices] * g / norm
            groups[group] = dict(parameter_labels=[parameter_labels[i] for i in indices],
                                 coordinate_scales=scales[indices].tolist(), scaled_gradient_l2=norm,
                                 scaled_root_gradient_l2=float(np.linalg.norm(gr)) if has_root else None,
                                 physical_descent_direction=physical_direction.tolist(),
                                 predicted_cost_change_per_unit_descent=-norm,
                                 predicted_root_change_per_unit_descent=rate,
                                 predicted_position_cost_change=error_rate,
                                 predicted_Er_at_target_change=er_rate,
                                 predicts_position_improvement=(error_rate < 0) if error_rate is not None else None)
            if inner_field_gradient is not None:
                groups[group]["predicted_inner_mean_Er_change_per_unit_descent"] = (
                    float(-np.dot(inner_field_gradient[indices] * scales[indices], g) / norm) if norm > 0 else None)
        comparisons.append(dict(name=name, cost=float(cost),
                                physical_parameter_gradient=dict(zip(parameter_labels, map(float, gradient), strict=True)),
                                groups=groups))
    result = dict(target=settings.target, annulus=[settings.inner_radius, settings.outer_radius],
                settings=dict(target=settings.target, inner_radius=settings.inner_radius, outer_radius=settings.outer_radius,
                              profile_trust_multiplier=settings.profile_trust_multiplier, geometry_ess_alpha=settings.geometry_ess_alpha),
                coordinate_scales=scales.tolist(), baseline_metrics=metrics, terminal_time=snapshot["final_time"],
                terminal_rho=rho.tolist(), terminal_Er=er.tolist(),
                actual_downcrossings=metrics["actual_downcrossings"], nearest_downcrossing=root,
                nearest_downcrossing_available=has_root, comparisons=comparisons,
                interpretation="Negative predicted_position_cost_change improves the selected real chord root locally. "
                               "Conditional on this bracket; not a finite-step or convergence guarantee. "
                               "Profiles use nominal scales times the profile trust multiplier; geometry uses ESS scales.")
    if "original_objectives" in metrics:
        result["original_objective_settings"] = original_objective_settings()
    return result
