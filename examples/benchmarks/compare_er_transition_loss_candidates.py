"""Compare terminal Er losses on prescribed arrays, without plasma solves.

These are audit candidates, not production objectives. All candidates are
calibrated once to the production loss on the same displaced-root profile;
the frozen factors make gradient comparisons less dependent on arbitrary
overall weights. No transport/profile-parameter Jacobian is inferred here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.benchmarks import audit_er_transition_surrogate_objectives as audit


NAMES = (
    "current",
    "capped_selector",
    "local_margins",
    "target_zero_local",
    "target_zero_regional",
)
TARGET = audit.example.ER_TRANSITION_RHO_TARGET
MARGIN = audit.example.ER_TRANSITION_STRENGTH_TARGET
ER_SCALE = audit.example.ER_TRANSITION_TEMPERATURE_KV_M


def grid():
    return (jnp.arange(51, dtype=jnp.float64) + 0.5) / 51


def capped_location(er, rho):
    """Stop rewarding extra face strength after the required margin is met.

    Only selection changes. The accompanying strength penalty remains the
    original global one, deliberately testing whether capping alone suffices.
    """
    inner, outer = er[:-1] / ER_SCALE, -er[1:] / ER_SCALE
    score = 0.5 * (inner + outer - jnp.sqrt((inner - outer) ** 2 + 1e-12))
    magnitude = jnp.sqrt(er**2 + (ER_SCALE * 1e-6) ** 2)
    candidates = rho[:-1] + jnp.diff(rho) * magnitude[:-1] / (magnitude[:-1] + magnitude[1:])
    bias = -0.5 * ((candidates - TARGET) / audit.example.ER_TRANSITION_RHO_SOFTNESS) ** 2
    faces = 0.5 * (rho[:-1] + rho[1:])
    mask = (faces >= audit.example.ER_TRANSITION_RHO_MIN) & (faces <= audit.example.ER_TRANSITION_RHO_MAX)
    mask = jnp.where(jnp.any(mask), mask, jnp.arange(faces.size) == jnp.argmin(jnp.abs(faces - TARGET)))
    cap_temperature = 0.25
    bounded_score = MARGIN - cap_temperature * jax.nn.softplus((MARGIN - score) / cap_temperature)
    logits = jnp.where(mask, bias + audit.example.ER_TRANSITION_SOFTMAX_BETA * bounded_score, -jnp.inf)
    return jnp.sum(jax.nn.softmax(logits) * candidates)


def target_zero_region_residuals(er, rho, *, rho_min, rho_max, rho_target=TARGET):
    """One root residual block: an interpolated zero and annular sign errors.

    Bounds are explicit audit settings. In particular, the lower bound need
    not include the core, where small negative/zero fields may be allowed.
    This does not search for the smallest |Er| or choose an axial zero.
    """
    if not rho_min < rho_target < rho_max:
        raise ValueError("The sign-constraint band must contain the target radius.")
    at_target = jnp.interp(rho_target, rho, er) / ER_SCALE
    mask = (rho >= rho_min) & (rho <= rho_max)
    half_cell = 0.5 * jnp.mean(jnp.diff(rho))
    signed_distance = (rho_target - rho) / jnp.sqrt((rho_target - rho)**2 + half_cell**2)
    deficits = audit.example.smooth_positive_part(MARGIN * signed_distance**2 - signed_distance * er / ER_SCALE)
    # The mean normalization removes dependence on the number of active
    # samples. All entries share one external least-squares weight; there
    # are no separate position/strength multipliers in this candidate.
    sign_residuals = jnp.where(mask, deficits / jnp.sqrt(jnp.sum(mask)), 0.0)
    return jnp.concatenate((jnp.atleast_1d(at_target), sign_residuals))


def target_zero_region_cost(er, rho, *, rho_min, rho_max, root_weight=1.0, rho_target=TARGET):
    residuals = target_zero_region_residuals(er, rho, rho_min=rho_min, rho_max=rho_max, rho_target=rho_target)
    return 0.5 * root_weight * jnp.vdot(residuals, residuals)


def losses(er, rho):
    """Return five costs before the once-only reference calibration.

    Local penalties ask for >=1.8 kV/m on the correct side. The regional
    margin tapers smoothly to zero at the target, so a grid node at the
    requested zero is not simultaneously asked for a nonzero field. Neither
    prescribes an amplitude above its margin. The regional average covers
    the existing optimization window, not the magnetic axis or the edge.
    """
    location, strength = audit.metrics(er, rho)
    current = 0.5 * (
        audit.example.ER_TRANSITION_RHO_WEIGHT * (location - TARGET) ** 2
        + audit.example.ER_TRANSITION_STRENGTH_WEIGHT * audit.strength_deficit(strength) ** 2
    )
    capped = 0.5 * (
        audit.example.ER_TRANSITION_RHO_WEIGHT * (capped_location(er, rho) - TARGET) ** 2
        + audit.example.ER_TRANSITION_STRENGTH_WEIGHT * audit.strength_deficit(strength) ** 2
    )
    left = jnp.clip(jnp.searchsorted(rho, TARGET, side="left") - 1, 0, rho.size - 2)
    right = jnp.clip(jnp.searchsorted(rho, TARGET, side="right"), 1, rho.size - 1)
    local_signed_er = jnp.stack((er[left], -er[right])) / ER_SCALE
    local = 0.5 * jnp.mean(audit.example.smooth_positive_part(MARGIN - local_signed_er) ** 2)
    at_target = jnp.interp(TARGET, rho, er) / ER_SCALE
    zero_cost = 0.5 * at_target**2
    regional = target_zero_region_cost(
        er, rho, rho_min=audit.example.ER_TRANSITION_RHO_MIN,
        rho_max=audit.example.ER_TRANSITION_RHO_MAX,
    )
    return jnp.stack((current, capped, local, zero_cost + local, regional))


def profile_with_free_core(root, core_amplitude):
    """Known off-grid crossing plus an independently varied small core field.

    The core oscillations and their blend end by rho=0.28. The annular probe
    starts at 0.30; that cutoff is illustrative, not a production prescription.
    """
    rho = grid()
    fraction = jnp.clip((rho - 0.20) / 0.08, 0.0, 1.0)
    blend = fraction**2 * (3.0 - 2.0 * fraction)
    core = core_amplitude * jnp.sin(4 * jnp.pi * rho / 0.25)
    crossing = jnp.clip(500.0 * (root - rho), -10.0, 10.0)
    return blend * crossing + (1.0 - blend) * core


def axis_offgrid_probe():
    """Check core invariance while retaining response to the desired root."""
    rho = grid()
    inner, outer = 0.30, audit.example.ER_TRANSITION_RHO_MAX

    def cost(root, core):
        return target_zero_region_cost(profile_with_free_core(root, core), rho, rho_min=inner, rho_max=outer)

    evaluate = jax.jit(jax.value_and_grad(cost, argnums=(0, 1)))
    rows = []
    for root in (TARGET - 0.008, TARGET, TARGET + 0.008):
        for core in (-0.2, 0.0, 0.2):
            er = profile_with_free_core(root, core)
            value, (root_bar, core_bar) = evaluate(root, core)
            er_bars = jax.grad(lambda values: target_zero_region_cost(values, rho, rho_min=inner, rho_max=outer))(er)
            rows.append(dict(
                true_transition_radius=root, target_radius=TARGET,
                target_is_grid_node=bool(jnp.any(rho == TARGET)), core_amplitude=core,
                inner_constraint_radius=inner, outer_constraint_radius=outer,
                Er_target=float(jnp.interp(TARGET, rho, er)),
                Er_left=float(er[25]), Er_right=float(er[26]),
                annular_cost=float(value), d_annular_cost_d_root=float(root_bar),
                d_annular_cost_d_core=float(core_bar),
                max_abs_core_Er_bar=float(jnp.max(jnp.where(rho < 0.28, jnp.abs(er_bars), 0.0))),
                original_regional_cost=float(target_zero_region_cost(
                    er, rho, rho_min=audit.example.ER_TRANSITION_RHO_MIN, rho_max=outer,
                )),
            ))
    return rows


def one_weight_probe():
    """Verify the same single weight scales the cost and root direction."""
    rho = grid()
    rows = []
    for weight in (0.0, 0.1, 1.0, 10.0):
        objective = lambda root, core: target_zero_region_cost(
            profile_with_free_core(root, core), rho, rho_min=0.30, rho_max=0.9,
            root_weight=weight,
        )
        value, (root_bar, core_bar) = jax.value_and_grad(objective, argnums=(0, 1))(TARGET + 0.008, -0.2)
        rows.append(dict(root_weight=weight, cost=float(value),
                         d_cost_d_root=float(root_bar), d_cost_d_core=float(core_bar)))
    return rows


def calibration():
    rho = grid()
    reference = 10.0 * jnp.tanh((0.65 - rho) / 0.01)
    raw = np.asarray(losses(reference, rho))
    return raw[0] / raw


def scenarios():
    """Return (sweep, coordinate, parameter, rho, Er, dEr/dparameter)."""
    rho = grid()
    cases = []

    def add(sweep, coordinate, parameter, function, mesh=rho):
        cases.append((sweep, coordinate, parameter, mesh, function(parameter), jax.jacfwd(function)(parameter)))

    for q in (0.35, 0.45, TARGET, 0.58, 0.65):
        add("translate_resolved", "q", q, lambda value: 10.0 * jnp.tanh((value - rho) / 0.02))
    for width in (0.05, 0.02, 0.005, 0.001):
        add(f"translate_width_{width:g}", "q", float(26 / 51),
            lambda value, width=width: 40.0 * jnp.tanh((value - rho) / width))
    remote = 10.0 * jnp.tanh((0.65 - rho) / 0.01)
    for p in (0.0, 5.0, 9.0, 10.0, 12.0, 14.0, 20.0):
        add("build_target_bracket", "p", p, lambda value: remote.at[26].add(-value))
    local_crossing = remote.at[26].add(-14.0)
    repair_direction = jnp.where((rho > rho[26]) & (rho < 0.65), -2.0 * jnp.maximum(remote, 0.0), 0.0)
    for p in (0.0, 0.25, 0.5, 0.75, 1.0):
        add("repair_outer_positive_pocket", "p", p, lambda value: local_crossing + value * repair_direction)
    for offset in (-10.0, 0.0, 10.0):
        add(f"missing_crossing_offset_{offset:g}", "p", 0.0,
            lambda value, offset=offset: jnp.full_like(rho, offset) + value * jnp.tanh((TARGET - rho) / 0.02))
    # A linear target crossing is exactly represented by interpolation. Above
    # the margin, increasing its amplitude should not be necessary.
    for amplitude in (0.0, 0.1, 0.5, 1.0, 2.0):
        add("amplitude_at_correct_root", "amplitude", amplitude, lambda value: value * 500.0 * (TARGET - rho))
    for amplitude in (0.0, 0.05, 0.1, 0.25, 1.0, 2.0):
        add("amplitude_at_wrong_root", "amplitude", amplitude, lambda value: value * remote)
    add("wrong_orientation", "positive_amplitude", 1.0,
        lambda value: -value * 10.0 * jnp.tanh((TARGET - rho) / 0.02))
    node_grid = rho.at[25].set(TARGET)
    add("target_on_grid_node", "amplitude", 1.0,
        lambda value: value * 1000.0 * (TARGET - node_grid), node_grid)
    # Both downcrossings are far from the target; an upward crossing is at the
    # target. Their weighted mean is exactly the target. This four-node case
    # isolates the false-centroid failure; the other cases use the 51-cell grid.
    false_grid = jnp.asarray([TARGET - 0.13, TARGET - 0.11, TARGET + 0.11, TARGET + 0.13])
    add("false_centroid", "amplitude", 1.0,
        lambda value: value * jnp.asarray([10.0, -10.0, 10.0, -10.0]), false_grid)
    return cases


def profile_metrics(er, rho):
    er, rho = np.asarray(er), np.asarray(rho)
    faces = 0.5 * (rho[:-1] + rho[1:])
    mask = (faces >= audit.example.ER_TRANSITION_RHO_MIN) & (faces <= audit.example.ER_TRANSITION_RHO_MAX)
    crossings = (er[:-1] >= 0.0) & (er[1:] < 0.0) & mask
    roots = rho[:-1][crossings] + np.diff(rho)[crossings] * er[:-1][crossings] / (er[:-1][crossings] - er[1:][crossings])
    left = int(np.clip(np.searchsorted(rho, TARGET, side="left") - 1, 0, rho.size - 2))
    right = int(np.clip(np.searchsorted(rho, TARGET, side="right"), 1, rho.size - 1))
    sign_mask = (rho >= audit.example.ER_TRANSITION_RHO_MIN) & (rho <= audit.example.ER_TRANSITION_RHO_MAX)
    wrong = np.where(rho < TARGET, er < 0, er > 0) & sign_mask
    return dict(
        downcrossing_count=int(crossings.sum()),
        nearest_downcrossing_error=float(np.min(np.abs(roots - TARGET))) if len(roots) else "",
        Er_target=float(np.interp(TARGET, rho, er)),
        Er_target_left=float(er[left]), Er_target_right=float(er[right]),
        wrong_sign_fraction=float(wrong.sum() / sign_mask.sum()),
    )


def compare_rows(factors):
    values_and_bars = jax.jit(lambda er, rho: (losses(er, rho), jax.jacrev(losses)(er, rho)))
    rows = []
    for sweep, coordinate, parameter, rho, er, direction in scenarios():
        values, bars = values_and_bars(er, rho)
        values, bars = np.asarray(values) * factors, np.asarray(bars) * factors[:, None]
        derivatives = bars @ np.asarray(direction)
        # The production smooth-min has an Er smoothing scale of 2e-6.
        # At the identically-zero profile, a generic 1e-5 parameter step
        # crosses that scale and is not a local derivative check.
        step = 1e-6 if np.any(np.asarray(er)) else 1e-8 / max(1.0, float(jnp.max(jnp.abs(direction))))
        fd = factors * np.asarray((losses(er + step * direction, rho) - losses(er - step * direction, rho)) / (2 * step))
        physical = profile_metrics(er, rho)
        for i, name in enumerate(NAMES):
            rows.append(dict(
                sweep=sweep, coordinate=coordinate, parameter=parameter, candidate=name,
                cost=float(values[i]), gradient=float(derivatives[i]),
                directional_fd=float(fd[i]), fd_absolute_error=float(abs(fd[i] - derivatives[i])),
                fd_scaled_error=float(abs(fd[i] - derivatives[i]) / max(1.0, abs(fd[i]), abs(derivatives[i]))),
                fd_parameter_step=step,
                dEr_direction_l2=float(jnp.linalg.norm(direction)),
                Er_gradient_l2=float(np.linalg.norm(bars[i])), **physical,
            ))
    return rows


def scalar_search_rows(factors):
    """One-dimensional loss landscapes, not optimization of a plasma model.

    Use a dense fixed grid to reveal flat regions and competing minima without
    introducing an optimizer-specific scaling or line-search explanation.
    """
    rho = grid()
    remote = 10.0 * jnp.tanh((0.65 - rho) / 0.01)
    scans = (
        ("translate_resolved", "q", np.linspace(0.35, 0.7, 351),
         lambda q: 10.0 * jnp.tanh((q - rho) / 0.02)),
        ("build_target_bracket", "p", np.linspace(0.0, 24.0, 241),
         lambda p: remote.at[26].add(-p)),
    )
    rows = []
    for sweep, coordinate, parameters, function in scans:
        matrix = np.asarray(jax.jit(jax.vmap(lambda p: losses(function(p), rho)))(jnp.asarray(parameters))) * factors
        for p, values in zip(parameters, matrix, strict=True):
            for name, value in zip(NAMES, values, strict=True):
                rows.append(dict(sweep=sweep, coordinate=coordinate, parameter=float(p), candidate=name, cost=float(value)))
    return rows


def plot_comparison(out_dir, rows, scans):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    labels = ("Current surrogate", "Capped selector", "Local margins",
              "Target zero + local margins", "Target zero + regional margins")
    for name, label in zip(NAMES, labels, strict=True):
        selected = [r for r in scans if r["sweep"] == "translate_resolved" and r["candidate"] == name]
        axes[0, 0].plot([r["parameter"] for r in selected], [max(r["cost"], 1e-10) for r in selected], label=label)
        selected = [r for r in scans if r["sweep"] == "build_target_bracket" and r["candidate"] == name]
        axes[0, 1].plot([r["parameter"] for r in selected], [max(r["cost"], 1e-10) for r in selected], label=label)
        selected = [r for r in rows if r["sweep"] == "build_target_bracket" and r["candidate"] == name]
        axes[1, 0].plot([r["parameter"] for r in selected], [r["gradient"] for r in selected], "o-", label=label)
        selected = [r for r in rows if r["sweep"] == "repair_outer_positive_pocket" and r["candidate"] == name]
        axes[1, 1].plot([r["parameter"] for r in selected], [r["cost"] for r in selected], "o-", label=label)
    axes[0, 0].axvline(TARGET, color="grey", linestyle="--", linewidth=1)
    axes[0, 0].set(xlabel="True crossing q", ylabel="Calibrated cost", yscale="log", title="Move a resolved crossing")
    axes[0, 1].set(xlabel="Lower outer target cell by p (kV/m)", ylabel="Calibrated cost", yscale="log", title="Build crossing beside a strong remote root")
    axes[1, 0].axhline(0.0, color="grey", linewidth=1)
    axes[1, 0].set(xlabel="p (kV/m)", ylabel="d(cost)/dp", title="Negative gradient encourages the new crossing")
    axes[1, 1].set(xlabel="Fraction of outer positive-pocket repair", ylabel="Calibrated cost", title="Remove extra roots after local crossing exists")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Prescribed Er arrays only; no transport or geometry solve", fontsize=12)
    fig.savefig(out_dir / "candidate_comparison.svg")
    fig.savefig(out_dir / "candidate_comparison.png", dpi=120)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/er_transition_loss_candidates")
    args = parser.parse_args(argv)
    # NEOPAX import can enable its cache after reading environment variables.
    # Disable it here for this small audit without affecting production imports.
    jax.config.update("jax_enable_compilation_cache", False)
    factors = calibration()
    rows = compare_rows(factors)
    scans = scalar_search_rows(factors)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit.write_csv(args.out_dir / "candidates.csv", rows)
    audit.write_csv(args.out_dir / "landscapes.csv", scans)
    audit.write_csv(args.out_dir / "calibration.csv", [dict(candidate=name, factor=float(factor)) for name, factor in zip(NAMES, factors, strict=True)])
    audit.write_csv(args.out_dir / "axis_offgrid.csv", axis_offgrid_probe())
    audit.write_csv(args.out_dir / "one_weight_scaling.csv", one_weight_probe())
    plot_comparison(args.out_dir, rows, scans)
    print(json.dumps(dict(
        out_dir=str(args.out_dir), prescribed_profiles=len(rows) // len(NAMES),
        max_directional_fd_absolute_error=max(r["fd_absolute_error"] for r in rows),
        max_directional_fd_scaled_error=max(r["fd_scaled_error"] for r in rows),
        calibration=dict(zip(NAMES, map(float, factors), strict=True)),
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
