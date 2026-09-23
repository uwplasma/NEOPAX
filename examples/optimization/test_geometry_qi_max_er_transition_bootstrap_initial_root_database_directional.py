#!/usr/bin/env python
"""Directional value/Jacobian check for live-database initial-root AD.

The check is run at the baseline and at a nonzero VMEC boundary vector.  The
second point is essential: a root evaluator that accidentally retained the
baseline database can appear correct at ``x0`` while returning stale transport
values after an optimizer update.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from NEOPAX import _reverse_ad_initial_er as initial_er_reverse  # noqa: E402
from NEOPAX import _reverse_ad_optimization as reverse_optimization  # noqa: E402
import optimize_geometry_qi_max_er_transition_bootstrap_initial_root as base  # noqa: E402


DATABASE_TRANSPORT_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_black_box.toml"
)
SMALL_DATABASE_TRANSPORT_CONFIG = (
    ROOT
    / "examples"
    / "benchmarks"
    / "Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_black_box_small.toml"
)


def _terms_for_objective_set(objective_set: str):
    """Keep the FD oracle focused on one selected-root derivative boundary."""

    selected = []
    for term in base.terms:
        objective = getattr(term[0], "objective", term[0])
        family = getattr(objective, "family", None)
        is_transport = family == "transport"
        is_bootstrap = (
            is_transport
            and objective.name == "bootstrap_current_softmax_abs_scaled"
        )
        if objective_set == "transport_er_only" and (not is_transport or is_bootstrap):
            continue
        if objective_set == "bootstrap_only" and not is_bootstrap:
            continue
        if objective_set == "geometry_only" and family != "geometry":
            continue
        selected.append(term)
    if not selected:
        raise RuntimeError(f"No terms selected for objective_set={objective_set!r}.")
    return selected


def _evaluation_arrays(evaluation):
    return (
        np.asarray(jax.device_get(evaluation.residuals), dtype=float),
        np.asarray(jax.device_get(evaluation.jacobian), dtype=float),
        tuple(evaluation.result.residual_labels),
    )


def _evaluate_with_selected_root_capture(problem, x):
    """Evaluate AD once and retain the central selected-root linearization data."""

    original_selected_root = reverse_optimization.initial_er_selected_root_profile
    captured = {}

    def _capture_selected_root(state, *, config, runtime):
        er_profile, finite_mask = original_selected_root(state, config=config, runtime=runtime)
        captured["state"] = state
        captured["runtime"] = runtime
        captured["er_profile"] = er_profile
        captured["finite_mask"] = finite_mask
        return er_profile, finite_mask

    with patch.object(reverse_optimization, "initial_er_selected_root_profile", _capture_selected_root):
        evaluation = problem.evaluate(x)
    if set(captured) != {"state", "runtime", "er_profile", "finite_mask"}:
        raise RuntimeError("The central database evaluation did not expose its selected ambipolar root.")
    baseline_residual = initial_er_reverse.initial_er_charge_flux_residuals(
        captured["state"], captured["er_profile"], runtime=captured["runtime"]
    )
    baseline_dres_der = initial_er_reverse.initial_er_charge_flux_residual_er_derivative(
        captured["state"], captured["er_profile"], runtime=captured["runtime"]
    )
    captured["baseline_residual"] = jax.block_until_ready(baseline_residual)
    captured["baseline_dres_der"] = jax.block_until_ready(baseline_dres_der)
    return _evaluation_arrays(evaluation), captured


def _evaluate_with_frozen_linearized_root(problem, x, root_data):
    """Replay one FD endpoint with the central implicit Er-root branch frozen."""

    baseline_er = root_data["er_profile"]
    baseline_residual = root_data["baseline_residual"]
    baseline_dres_der = root_data["baseline_dres_der"]
    baseline_mask = root_data["finite_mask"]

    def _frozen_selected_root(state, *, config, runtime):
        del config
        er0 = jnp.asarray(baseline_er, dtype=state.Er.dtype)
        residual = initial_er_reverse.initial_er_charge_flux_residuals(state, er0, runtime=runtime)
        residual_delta = residual - jnp.asarray(baseline_residual, dtype=residual.dtype)
        dres_der = jnp.asarray(baseline_dres_der, dtype=residual.dtype)
        safe_dres_der = jnp.where(
            jnp.abs(dres_der) > jnp.asarray(1.0e-30, dtype=residual.dtype), dres_der, jnp.inf
        )
        er_profile = er0 - residual_delta / safe_dres_der
        return er_profile, jnp.asarray(baseline_mask, dtype=bool)

    with patch.object(reverse_optimization, "initial_er_selected_root_profile", _frozen_selected_root):
        return _evaluation_arrays(problem.evaluate(x))


def _check_point(problem, x, *, parameter_index: int, fd_step: float, label: str, objective_set: str):
    (residuals, jacobian, labels), root_data = _evaluate_with_selected_root_capture(problem, x)
    x_plus = np.array(x, copy=True)
    x_minus = np.array(x, copy=True)
    x_plus[parameter_index] += fd_step
    x_minus[parameter_index] -= fd_step
    residuals_plus, _, _ = _evaluate_with_frozen_linearized_root(problem, x_plus, root_data)
    residuals_minus, _, _ = _evaluate_with_frozen_linearized_root(problem, x_minus, root_data)
    finite_difference = (residuals_plus - residuals_minus) / (2.0 * fd_step)
    ad_column = jacobian[:, parameter_index]
    difference = ad_column - finite_difference
    denominator = np.maximum(np.maximum(np.abs(ad_column), np.abs(finite_difference)), 1.0e-12)
    relative = np.abs(difference) / denominator
    transport_rows = [
        row
        for row, name in enumerate(labels)
        if any(
            marker in str(name)
            for marker in ("softmax_Er", "Er_transition", "bootstrap_current")
        )
    ]
    if objective_set != "geometry_only" and not transport_rows:
        raise RuntimeError("The database directional check did not find any transport residual rows.")
    transport_shift = (
        0.0
        if not transport_rows
        else float(np.max(np.abs(residuals_plus[transport_rows] - residuals_minus[transport_rows])))
    )
    max_row = int(np.argmax(np.abs(difference)))
    transport_difference = difference[transport_rows]
    transport_relative = relative[transport_rows]
    transport_max_row = (
        None
        if not transport_rows
        else transport_rows[int(np.argmax(np.abs(transport_difference)))]
    )
    print(
        f"[database directional] point={label} objective_set={objective_set} "
        "fd_root_lane=frozen_linearized "
        f"residual_norm={np.linalg.norm(residuals):.6e} "
        f"fd_jacobian_max_abs={np.max(np.abs(difference)):.6e} "
        f"fd_jacobian_max_relative={np.max(relative):.6e} "
        f"fd_max_row={labels[max_row]!s} "
        f"fd_max_row_ad={ad_column[max_row]:.6e} "
        f"fd_max_row_fd={finite_difference[max_row]:.6e} "
        f"transport_fd_max_abs={0.0 if not transport_rows else np.max(np.abs(transport_difference)):.6e} "
        f"transport_fd_max_relative={0.0 if not transport_rows else np.max(transport_relative):.6e} "
        f"transport_fd_max_row={'none' if transport_max_row is None else labels[transport_max_row]!s} "
        f"transport_fd_max_row_ad={0.0 if transport_max_row is None else ad_column[transport_max_row]:.6e} "
        f"transport_fd_max_row_fd={0.0 if transport_max_row is None else finite_difference[transport_max_row]:.6e} "
        f"transport_residual_two_sided_shift={transport_shift:.6e}",
        flush=True,
    )
    if objective_set != "geometry_only" and transport_shift == 0.0:
        raise AssertionError(
            "Database transport residuals did not change under a boundary perturbation; "
            "the evaluator may be using a stale baseline database."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameter-index", type=int, default=0)
    parser.add_argument("--fd-step", type=float, default=1.0e-5)
    parser.add_argument("--base-offset", type=float, default=1.0e-3)
    parser.add_argument(
        "--objective-set",
        choices=("all", "transport_er_only", "bootstrap_only", "geometry_only"),
        default="all",
        help=(
            "Check all terms, selected-Er transport terms, bootstrap alone, "
            "or geometry alone."
        ),
    )
    parser.add_argument(
        "--small-database",
        action="store_true",
        help="Use the reduced (5, 25, 31) theta/zeta/xi NTX grid.",
    )
    parser.add_argument(
        "--include-x0",
        action="store_true",
        help="Also run the baseline x0 central-difference check (three additional evaluations).",
    )
    args = parser.parse_args()
    if args.fd_step <= 0.0:
        raise ValueError("--fd-step must be positive.")
    if not np.isscalar(base.MAX_MODE_SCHEDULE):
        raise ValueError("This directional check requires one fixed MAX_MODE_SCHEDULE value.")

    base.TRANSPORT_CONFIG = (
        SMALL_DATABASE_TRANSPORT_CONFIG if args.small_database else DATABASE_TRANSPORT_CONFIG
    )
    base.REVERSE_STAGE_MODE = "database"
    base.terms = _terms_for_objective_set(args.objective_set)
    problem = base.build_transition_bootstrap_initial_root_problem(
        base.SEED_INPUT, int(base.MAX_MODE_SCHEDULE)
    )
    x0 = np.asarray(jax.device_get(problem.x0), dtype=float)
    if not 0 <= args.parameter_index < x0.size:
        raise ValueError(f"--parameter-index must be in [0, {x0.size}); got {args.parameter_index}.")
    x_perturbed = np.array(x0, copy=True)
    x_perturbed[args.parameter_index] += args.base_offset
    if args.include_x0:
        _check_point(
            problem,
            x0,
            parameter_index=args.parameter_index,
            fd_step=args.fd_step,
            label="x0",
            objective_set=args.objective_set,
        )
    _check_point(
        problem,
        x_perturbed,
        parameter_index=args.parameter_index,
        fd_step=args.fd_step,
        label="perturbed",
        objective_set=args.objective_set,
    )
    print("[database directional] complete: transport values respond at the perturbed point.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
