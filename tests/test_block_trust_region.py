from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import NEOPAX._block_trust_region as block_trust_region
from NEOPAX._block_trust_region import block_trust_region_least_squares


jax.config.update("jax_enable_x64", True)


class _LinearProblem:
    def __init__(self, matrix, target):
        self.matrix = np.asarray(matrix, dtype=float)
        self.target = np.asarray(target, dtype=float)
        self.x0 = jnp.zeros((self.matrix.shape[1],), dtype=jnp.float64)
        self.points = []

    def evaluate(self, values):
        host = np.asarray(values, dtype=float)
        self.points.append(host.copy())
        return SimpleNamespace(
            residuals=jnp.asarray(self.matrix @ host - self.target),
            jacobian=jnp.asarray(self.matrix),
            elapsed_s=0.0,
        )


def test_block_trust_region_converges_with_separate_physical_limits():
    matrix = np.asarray(
        (
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (1.0, -1.0, 0.5),
            (0.2, 0.3, -0.4),
        )
    )
    target = np.asarray((1.0, -0.5, 0.2, 0.1))
    problem = _LinearProblem(matrix, target)
    result = block_trust_region_least_squares(
        problem,
        profile_mask=np.asarray((True, False, False)),
        bounds=(
            np.asarray((-1.0, -np.inf, -np.inf)),
            np.asarray((1.0, np.inf, np.inf)),
        ),
        max_nfev=20,
        geometry_initial_radius=1.0,
        geometry_min_radius=1.0,
        geometry_max_radius=1.0,
        profile_initial_fraction_limit=0.10,
        profile_min_fraction_limit=0.10,
        profile_max_fraction_limit=0.10,
        verbose=0,
    )
    reference = np.linalg.lstsq(matrix, target, rcond=None)[0]
    reference_cost = 0.5 * np.linalg.norm(matrix @ reference - target) ** 2
    assert result.cost == pytest.approx(reference_cost, abs=1.0e-10)
    assert result.accepted_steps > 0
    assert np.array_equal(np.asarray(problem.x0), np.zeros((3,)))

    evaluated = np.asarray(problem.points)
    trial_steps = np.diff(evaluated, axis=0)
    assert np.all(np.abs(trial_steps[:, 0]) <= 0.10 + 1.0e-10)
    assert np.all(np.linalg.norm(trial_steps[:, 1:], axis=1) <= 1.0 + 1.0e-10)


def test_block_trust_region_rejects_noncentered_initial_point_outside_bounds():
    problem = _LinearProblem(np.eye(2), np.ones((2,)))
    problem.x0 = jnp.asarray((2.0, 0.0), dtype=jnp.float64)
    with pytest.raises(ValueError, match="Initial coordinates"):
        block_trust_region_least_squares(
            problem,
            profile_mask=np.asarray((True, False)),
            bounds=(np.asarray((-1.0, -1.0)), np.asarray((1.0, 1.0))),
            verbose=0,
        )


def test_block_model_step_falls_back_when_slsqp_returns_nonfinite_step(
    monkeypatch,
):
    def failed_minimize(*_args, **_kwargs):
        return SimpleNamespace(
            x=np.asarray((np.nan, np.nan)),
            success=False,
            status=8,
            message="Positive directional derivative for linesearch",
        )

    import scipy.optimize

    monkeypatch.setattr(scipy.optimize, "minimize", failed_minimize)
    residuals = np.asarray((2.0, -1.0))
    jacobian = np.eye(2)
    step, metadata = block_trust_region._block_model_step(
        residuals,
        jacobian,
        np.zeros((2,)),
        np.asarray((-1.0, -1.0)),
        np.asarray((1.0, 1.0)),
        np.asarray((True, False)),
        geometry_radius=0.25,
        profile_fraction_limit=0.10,
        proximal_weight=1.0e-10,
    )

    assert metadata["subproblem_solver"] == "cauchy_fallback"
    assert np.all(np.isfinite(step))
    assert abs(step[0]) <= 0.10 + 1.0e-12
    assert abs(step[1]) <= 0.25 + 1.0e-12
    assert block_trust_region._cost(residuals + jacobian @ step) < (
        block_trust_region._cost(residuals)
    )


def test_block_model_step_projects_infeasible_slsqp_descent(monkeypatch):
    def infeasible_minimize(*_args, **_kwargs):
        return SimpleNamespace(
            x=np.asarray((-10.0, 10.0)),
            success=False,
            status=8,
            message="Positive directional derivative for linesearch",
        )

    import scipy.optimize

    monkeypatch.setattr(scipy.optimize, "minimize", infeasible_minimize)
    residuals = np.asarray((1.0, -1.0))
    jacobian = np.eye(2)
    step, metadata = block_trust_region._block_model_step(
        residuals,
        jacobian,
        np.zeros((2,)),
        np.asarray((-1.0, -1.0)),
        np.asarray((1.0, 1.0)),
        np.asarray((True, False)),
        geometry_radius=0.25,
        profile_fraction_limit=0.10,
        proximal_weight=1.0e-10,
    )

    assert metadata["subproblem_solver"] == "slsqp_projected"
    assert abs(step[0]) <= 0.10 + 1.0e-12
    assert abs(step[1]) <= 0.25 + 1.0e-12
    assert block_trust_region._cost(residuals + jacobian @ step) < (
        block_trust_region._cost(residuals)
    )


def test_block_example_owns_cli_configurable_objective_weights():
    from examples.optimization import (
        optimize_geometry_profiles_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_block_trust_region
        as block_example,
    )

    args = block_example.parser().parse_args(
        ("--qi-weight", "2.5", "--net-power-weight", "7.0")
    )
    terms = block_example.active_terms(args)
    weights = {objective.label: weight for objective, _target, weight in terms}

    assert weights[block_example.combined_example.opt.geometry.boozer_qi_objective.label] == 2.5
    assert (
        weights[
            block_example.combined_example.opt.transport.net_total_power_volume_average_mw_m3.label
        ]
        == 7.0
    )
    assert block_example.combined_example.QI_WEIGHT != 2.5
    assert block_example.combined_example.NET_POWER_WEIGHT != 7.0
