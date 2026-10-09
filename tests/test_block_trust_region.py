from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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
