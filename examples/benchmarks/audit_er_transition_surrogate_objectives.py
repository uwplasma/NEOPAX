"""Probe Er-surrogate conditioning on short vectors; no plasma solves.

Uses the production transition formula and optimization weights. This audit
does not build a runtime, solve ambipolar roots, or integrate transport.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from NEOPAX._reverse_ad_transport import smooth_positive_to_negative_er_transition_metrics
from examples.optimization import (
    optimize_geometry_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport
    as example,
)


def metrics(er, rho, *, beta=None):
    return jnp.stack(smooth_positive_to_negative_er_transition_metrics(
        er, rho,
        rho_min=example.ER_TRANSITION_RHO_MIN,
        rho_max=example.ER_TRANSITION_RHO_MAX,
        rho_prior=example.ER_TRANSITION_RHO_TARGET,
        temperature_kv_m=example.ER_TRANSITION_TEMPERATURE_KV_M,
        positive_part_eps=1e-6,
        rho_softness=example.ER_TRANSITION_RHO_SOFTNESS,
        softmax_beta=example.ER_TRANSITION_SOFTMAX_BETA if beta is None else beta,
    ))


def fixed_cell_cost(er):
    return 0.5 * (
        example.ER_TRANSITION_LEFT_WEIGHT * (er[0] - example.ER_TRANSITION_LEFT_TARGET) ** 2
        + example.ER_TRANSITION_RIGHT_WEIGHT * (er[1] - example.ER_TRANSITION_RIGHT_TARGET) ** 2
    )


def strength_deficit(strength):
    return example.smooth_positive_part(example.ER_TRANSITION_STRENGTH_TARGET - strength)


def surrogate_cost(er, rho):
    location, strength = metrics(er, rho)
    return 0.5 * (
        example.ER_TRANSITION_RHO_WEIGHT * (location - example.ER_TRANSITION_RHO_TARGET) ** 2
        + example.ER_TRANSITION_STRENGTH_WEIGHT * strength_deficit(strength) ** 2
    )


def amplitude_probe():
    """Vary both bracket amplitudes while their interpolated zero stays fixed."""
    rho = jnp.asarray([0.5, 0.52], dtype=jnp.float64)
    bracket = jnp.asarray([
        example.ER_TRANSITION_LEFT_TARGET, example.ER_TRANSITION_RIGHT_TARGET,
    ], dtype=jnp.float64)
    rows = []
    for amplitude in (0.5, 1.0, 1.5, 2.0):
        er = amplitude * bracket
        location, strength = metrics(er, rho)
        old_grad = jax.grad(lambda a: fixed_cell_cost(a * bracket))(amplitude)
        new_grad = jax.grad(lambda a: surrogate_cost(a * bracket, rho))(amplitude)
        position_grad = jax.grad(lambda a: metrics(a * bracket, rho)[0])(amplitude)
        rows.append(dict(
            amplitude=amplitude, Er_left=float(er[0]), Er_right=float(er[1]),
            location=float(location), strength=float(strength),
            fixed_cell_cost=float(fixed_cell_cost(er)),
            surrogate_cost=float(surrogate_cost(er, rho)),
            d_fixed_cell_cost_d_amplitude=float(old_grad),
            d_surrogate_cost_d_amplitude=float(new_grad),
            d_location_d_amplitude=float(position_grad),
        ))
    return rows


def selection_probe():
    """Two existing brackets: show loss of face-selection sensitivity at large beta."""
    rho = jnp.asarray([0.45, 0.47, 0.55, 0.57], dtype=jnp.float64)

    def profile(amplitude):
        return jnp.asarray([10.0 * amplitude, -10.0 * amplitude, 8.0, -8.0])

    rows = []
    for beta in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        location, strength = metrics(profile(1.0), rho, beta=beta)
        derivative = jax.grad(lambda a: metrics(profile(a), rho, beta=beta)[0])(1.0)
        rows.append(dict(
            beta=beta, location=float(location), strength=float(strength),
            d_location_d_inner_bracket_amplitude=float(derivative),
        ))
    return rows


def translation_probe():
    """Move a known crossing and measure whether the surrogate follows it.

    The true zero is the parameter q of 40*tanh((q-rho)/width). The 51-cell
    centre grid matches the realtime geometry builder. Keeping the zero at a
    face centre gives an independent analytical chord-tracking factor.
    """
    rho = (jnp.arange(51, dtype=jnp.float64) + 0.5) / 51
    left_index, right_index = 25, 26
    centre = float(0.5 * (rho[left_index] + rho[right_index]))
    spacing = float(rho[right_index] - rho[left_index])
    rows = []
    for width in (0.05, 0.02, 0.01, 0.005, 0.002, 0.001):
        def profile(q):
            return 40.0 * jnp.tanh((q - rho) / width)

        er = profile(centre)
        location, strength = metrics(er, rho)
        er_direction = jax.jacfwd(profile)(centre)
        er_bars = jax.grad(lambda values: metrics(values, rho)[0])(er)
        location_derivative = jax.grad(lambda q: metrics(profile(q), rho)[0])(centre)
        loss_derivative = jax.grad(lambda q: surrogate_cost(profile(q), rho))(centre)
        reference_loss_derivative = example.ER_TRANSITION_RHO_WEIGHT * (
            centre - example.ER_TRANSITION_RHO_TARGET
        )
        argument = spacing / width
        rows.append(dict(
            width=width, cell_spacing=spacing, width_in_cells=width / spacing,
            true_root=centre, selected_root=float(location), strength=float(strength),
            expected_d_true_root_d_q=1.0,
            d_selected_root_d_q=float(location_derivative),
            analytical_chord_tracking_factor=float(argument / np.sinh(argument)),
            chain_rule_er_bar_dot_direction=float(jnp.dot(er_bars, er_direction)),
            d_Er_left_d_q=float(er_direction[left_index]),
            d_Er_right_d_q=float(er_direction[right_index]),
            d_surrogate_cost_d_q=float(loss_derivative),
            d_true_position_cost_d_q=float(reference_loss_derivative),
        ))
    return rows


def target_region_probe():
    """Improve the target-region sign pattern while a strong remote root exists.

    The comparison term is a local sign-margin diagnostic, not the original
    amplitude targets and not a replacement production objective. It tests
    whether the current loss responds before a new crossing wins selection.
    """
    rho = (jnp.arange(51, dtype=jnp.float64) + 0.5) / 51
    baseline = 10.0 * jnp.tanh((0.65 - rho) / 0.01)
    left_index, right_index = 25, 26
    margin = example.ER_TRANSITION_STRENGTH_TARGET * example.ER_TRANSITION_TEMPERATURE_KV_M

    def profile(p):
        return baseline.at[right_index].add(-p)

    def target_sign_cost(p):
        er = profile(p)
        left_deficit = example.smooth_positive_part(margin - er[left_index])
        right_deficit = example.smooth_positive_part(margin + er[right_index])
        return 0.5 * (left_deficit**2 + right_deficit**2)

    rows = []
    for p in (0.0, 5.0, 9.0, 10.0, 12.0, 14.0, 20.0):
        er = profile(p)
        location, strength = metrics(er, rho)
        bracket_exists = bool((er[left_index] > 0) & (er[right_index] < 0))
        target_root = (
            float(rho[left_index] + (rho[right_index] - rho[left_index])
                  * er[left_index] / (er[left_index] - er[right_index]))
            if bracket_exists else ""
        )
        rows.append(dict(
            p=p, Er_target_left=float(er[left_index]), Er_target_right=float(er[right_index]),
            target_bracket_exists=bracket_exists, target_bracket_root=target_root,
            target_bracket_margin=(
                float(jnp.minimum(er[left_index], -er[right_index])
                      / example.ER_TRANSITION_TEMPERATURE_KV_M)
                if bracket_exists else ""
            ),
            selected_root=float(location), strength=float(strength),
            surrogate_cost=float(surrogate_cost(er, rho)),
            d_surrogate_cost_d_p=float(jax.grad(lambda value: surrogate_cost(profile(value), rho))(p)),
            d_selected_root_d_p=float(jax.grad(lambda value: metrics(profile(value), rho)[0])(p)),
            target_sign_margin_cost=float(target_sign_cost(p)),
            d_target_sign_margin_cost_d_p=float(jax.grad(target_sign_cost)(p)),
        ))
    return rows


def strength_penalty_probe(strength, raw_parameter_derivative):
    deficit = strength_deficit(jnp.asarray(strength, dtype=jnp.float64))
    chain = jax.grad(strength_deficit)(jnp.asarray(strength, dtype=jnp.float64))
    weight = example.ER_TRANSITION_STRENGTH_WEIGHT
    return dict(
        strength=float(strength), threshold=example.ER_TRANSITION_STRENGTH_TARGET,
        raw_parameter_derivative=float(raw_parameter_derivative),
        deficit=float(deficit), d_deficit_d_strength=float(chain),
        weighted_residual=float(np.sqrt(weight) * deficit),
        d_weighted_residual_d_parameter=float(np.sqrt(weight) * chain * raw_parameter_derivative),
        weighted_cost=float(0.5 * weight * deficit * deficit),
        d_weighted_cost_d_parameter=float(weight * deficit * chain * raw_parameter_derivative),
    )


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "er_transition_surrogate_audit")
    parser.add_argument(
        "--benchmark-csv", type=Path,
        default=ROOT / "docs/benchmarks/database_transition_objectives_ad_recorded_reuse_full_t20_left_right_2026-10-22.csv",
    )
    args = parser.parse_args(argv)
    with args.benchmark_csv.open(encoding="utf-8", newline="") as stream:
        rows = {row["residual_label"]: row for row in csv.DictReader(stream)}
    strength = rows["transport:Er_transition_strength"]
    # Reweight the archived benchmark strength at its own point/prior. This
    # is not the Jacobian of a new optimizer run with a different prior.
    penalty = strength_penalty_probe(float(strength["residual"]), float(strength["n0"]))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "amplitude.csv", amplitude_probe())
    write_csv(args.out_dir / "selection.csv", selection_probe())
    write_csv(args.out_dir / "translation.csv", translation_probe())
    write_csv(args.out_dir / "target_region.csv", target_region_probe())
    write_csv(args.out_dir / "strength_penalty.csv", [penalty])
    print(json.dumps({"out_dir": str(args.out_dir), "archived_strength_penalty": penalty}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
