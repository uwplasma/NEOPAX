from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from NEOPAX._geometry_primary_trust_region import (
    geometry_primary_profile_correction_least_squares,
)


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


def test_geometry_is_solved_before_profile_correction():
    # Coordinate 0 is the profile delta and coordinate 1 is ESS geometry.
    matrix = np.asarray(((1.0, 1.0), (-1.0, 1.0)))
    target = np.asarray((0.2, 0.1))
    problem = _LinearProblem(matrix, target)

    result = geometry_primary_profile_correction_least_squares(
        problem,
        profile_mask=np.asarray((True, False)),
        bounds=(np.full((2,), -1.0), np.full((2,), 1.0)),
        max_nfev=2,
        geometry_initial_radius=1.0,
        geometry_min_radius=1.0,
        geometry_max_radius=1.0,
        profile_initial_fraction_limit=0.25,
        profile_min_fraction_limit=0.25,
        profile_max_fraction_limit=0.25,
        proximal_weight=0.0,
        verbose=0,
    )

    # The geometry-only fit gives dg=0.15.  With that value frozen, the
    # profile correction gives dp=0.05.  A joint or profile-first solve is not
    # used to redefine dg.
    assert problem.points[1][1] == pytest.approx(0.15, abs=1.0e-8)
    assert problem.points[1][0] == pytest.approx(0.05, abs=1.0e-8)
    assert result.accepted_steps == 1
    assert result.cost < 1.0e-14


class _RejectLargeProfileProblem(_LinearProblem):
    def evaluate(self, values):
        host = np.asarray(values, dtype=float)
        self.points.append(host.copy())
        if len(self.points) > 1 and abs(host[0]) > 0.05:
            raise RuntimeError("synthetic nonlinear profile failure")
        return SimpleNamespace(
            residuals=jnp.asarray(self.matrix @ host - self.target),
            jacobian=jnp.asarray(self.matrix),
            elapsed_s=0.0,
        )


def test_rejection_contracts_profile_before_geometry():
    problem = _RejectLargeProfileProblem(np.eye(2), np.asarray((0.2, 0.2)))
    result = geometry_primary_profile_correction_least_squares(
        problem,
        profile_mask=np.asarray((True, False)),
        bounds=(np.full((2,), -1.0), np.full((2,), 1.0)),
        max_nfev=3,
        geometry_initial_radius=1.0,
        geometry_min_radius=0.01,
        geometry_max_radius=1.0,
        profile_initial_fraction_limit=0.10,
        profile_min_fraction_limit=0.01,
        profile_max_fraction_limit=0.10,
        shrink_factor=0.25,
        proximal_weight=0.0,
        verbose=0,
    )

    assert result.rejected_steps == 1
    assert result.accepted_steps == 1
    assert result.profile_contractions == 1
    assert result.geometry_contractions == 0
    assert result.geometry_radius == pytest.approx(1.0)
    assert problem.points[1][0] == pytest.approx(0.10, abs=1.0e-8)
    assert problem.points[2][0] == pytest.approx(0.025, abs=1.0e-8)
    assert problem.points[1][1] == pytest.approx(0.20, abs=1.0e-8)
    assert problem.points[2][1] == pytest.approx(0.20, abs=1.0e-8)


class _RejectAnyProfileProblem(_LinearProblem):
    def evaluate(self, values):
        host = np.asarray(values, dtype=float)
        self.points.append(host.copy())
        if len(self.points) > 1 and host[0] != 0.0:
            raise RuntimeError("synthetic profile rejection")
        return SimpleNamespace(
            residuals=jnp.asarray(self.matrix @ host - self.target),
            jacobian=jnp.asarray(self.matrix),
            elapsed_s=0.0,
        )


def test_minimum_profile_rejection_tries_exact_geometry_step_next():
    problem = _RejectAnyProfileProblem(np.eye(2), np.asarray((0.2, 0.2)))
    result = geometry_primary_profile_correction_least_squares(
        problem,
        profile_mask=np.asarray((True, False)),
        bounds=(np.full((2,), -1.0), np.full((2,), 1.0)),
        max_nfev=3,
        geometry_initial_radius=1.0,
        geometry_min_radius=0.01,
        geometry_max_radius=1.0,
        profile_initial_fraction_limit=0.10,
        profile_min_fraction_limit=0.10,
        profile_max_fraction_limit=0.10,
        proximal_weight=0.0,
        verbose=0,
    )

    assert result.rejected_steps == 1
    assert result.accepted_steps == 1
    assert result.geometry_only_trials == 1
    assert result.geometry_contractions == 0
    assert problem.points[1] == pytest.approx((0.10, 0.20), abs=1.0e-8)
    assert problem.points[2] == pytest.approx((0.0, 0.20), abs=1.0e-8)


def test_solver_rejects_inconsistent_trust_settings_before_evaluation():
    problem = _LinearProblem(np.eye(2), np.ones((2,)))
    with pytest.raises(ValueError, match="Geometry trust radii must be ordered"):
        geometry_primary_profile_correction_least_squares(
            problem,
            profile_mask=np.asarray((True, False)),
            bounds=(np.full((2,), -1.0), np.full((2,), 1.0)),
            geometry_initial_radius=2.0,
            geometry_max_radius=1.0,
            verbose=0,
        )
    assert problem.points == []


def test_geometry_primary_example_has_its_own_output_lane_and_weights():
    from examples.optimization import (
        optimize_geometry_profiles_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_geometry_primary
        as example,
    )

    args = example.parser().parse_args(
        ("--qi-weight", "2.5", "--net-power-weight", "7.0")
    )
    assert args.out_dir == example.OUT_DIR
    assert args.profile_dofs is True
    assert args.qi_weight == pytest.approx(2.5)
    assert args.net_power_weight == pytest.approx(7.0)


def test_geometry_primary_problem_settings_extend_geometry_settings_only():
    from examples.optimization import (
        optimize_geometry_profiles_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_geometry_primary
        as example,
    )

    args = example.parser().parse_args(())
    kwargs = example._problem_kwargs(args, physical_pitches=None)
    geometry = example.combined_example.geometry_example

    assert kwargs["families"] == geometry.GEOMETRY_FAMILIES
    assert kwargs["scale_mode"] == geometry.SCALE_MODE == "ess"
    assert kwargs["ess_alpha"] == geometry.ESS_ALPHA
    assert kwargs["mboz"] == geometry.QI_MBOZ
    assert kwargs["nboz"] == geometry.QI_NBOZ
    assert kwargs["n_theta"] == geometry.DATABASE_N_THETA
    assert kwargs["n_zeta"] == geometry.DATABASE_N_PHI
    assert kwargs["n_xi"] == geometry.DATABASE_N_XI
    assert kwargs["reverse_stage_mode"] == geometry.REVERSE_STAGE_MODE
    assert kwargs["include_profiles"] is True
    assert kwargs["profile_scale_mode"] == "nominal"
    assert kwargs["profile_coordinate_mode"] == "delta"


def test_geometry_primary_example_rejects_negative_objective_weight():
    from examples.optimization import (
        optimize_geometry_profiles_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_geometry_primary
        as example,
    )

    args = example.parser().parse_args(("--net-power-weight", "-1"))
    with pytest.raises(ValueError, match="weights must be nonnegative"):
        example._validate_args(args)
