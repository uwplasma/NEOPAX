"""Exact, call-local database stage-Jacobian reuse and its dispatch contract."""

import dataclasses
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from NEOPAX import _transport_solvers as solvers


def _case():
    # Three mutually coupled stages, four coupled state variables, ten rows.
    dtype = jnp.float64
    sqrt6 = np.sqrt(6.0)
    a = jnp.asarray([
        [(88 - 7 * sqrt6) / 360, (296 - 169 * sqrt6) / 1800, (-2 + 3 * sqrt6) / 225],
        [(296 + 169 * sqrt6) / 1800, (88 + 7 * sqrt6) / 360, (-2 - 3 * sqrt6) / 225],
        [(16 - sqrt6) / 36, (16 + sqrt6) / 36, 1 / 9],
    ], dtype=dtype)
    kernel = SimpleNamespace(
        dtype=dtype, num_stages=3, state_dim=4, a=a, b=a[-1],
        c=jnp.asarray([(4 - sqrt6) / 10, (4 + sqrt6) / 10, 1], dtype=dtype),
        use_transport_lagged_response=False,
    )
    coupling = jnp.asarray([
        [-0.7, 0.3, -0.2, 0.1], [0.2, -0.6, 0.4, -0.3],
        [0.1, -0.4, -0.8, 0.2], [-0.2, 0.1, 0.3, -0.5],
    ], dtype=dtype)

    def rhs(t, y):
        return coupling @ jnp.sin(y) + 0.05 * y * jnp.sum(y ** 2) + t * jnp.exp(0.1 * y)

    def compact_state_must_not_run(*args, **kwargs):
        raise AssertionError("The exact block path must retain its finite jacfwd state derivative.")

    def support_pullback(t, y, rhs_bar, support):
        return {
            "database": support["database"] * jnp.dot(jnp.cos(y), rhs_bar),
            "geometry": (1 + t) * jnp.sum(rhs_bar) * jnp.ones_like(support["geometry"]),
        }

    physics = solvers._RadauAcceptedStepPhysicsContext(
        unpack_flat=lambda value: value, pack_flat=lambda value: value,
        project_flat=None, build_lagged_response=None,
        pullback_build_lagged_response=None, flat_rhs=rhs,
        flat_rhs_with_lagged_response=None, reverse_direct_stage_adjoint=True,
        reverse_stage_adjoint_solve_mode="block", reverse_rhs_transpose_mode="explicit_database",
        reverse_database_table_only=True, reverse_database_include_direct_geometry=True,
        flat_rhs_direct_black_box_state_pullback=compact_state_must_not_run,
        flat_rhs_direct_database_split_support_pullback=support_pullback,
    )
    y = jnp.asarray([0.3, -0.2, 0.7, 0.1], dtype=dtype)
    stages = jnp.linspace(-0.4, 0.6, 12, dtype=dtype).reshape((3, 4))
    dt = jnp.asarray(0.07, dtype=dtype)
    zero = jnp.asarray(0.0, dtype=dtype)
    zero_int = jnp.asarray(0, dtype=jnp.int32)
    carry = solvers._RadauAcceptedStepCarry(
        t=jnp.asarray(0.13, dtype=dtype), y=y, dt=dt, prev_error=zero,
        prev_stages=stages, prev_dt=dt, recent_reject_count=zero_int,
        regrowth_cooldown=zero_int, easy_growth_streak=zero_int,
        lagged_response_cache=None, lagged_response_valid=jnp.asarray(False),
        lagged_reference_y=y, jacobian=jnp.zeros((4, 4), dtype=dtype),
        cache_valid=jnp.asarray(False), cache_dt=dt, cache_age=zero_int,
        real_lu=jnp.eye(4, dtype=dtype), real_piv=jnp.zeros((4,), dtype=jnp.int32),
        complex_lu=jnp.eye(8, dtype=dtype), complex_piv=jnp.zeros((8,), dtype=jnp.int32),
        prev_theta_final=zero, prev_newton_iter_count=zero_int,
    )
    primal = solvers._RadauAcceptedStepReverseMinimalAttemptResult(
        carry_after_attempt=carry, trial_dt=dt, trial_y=y + dt * (kernel.b @ stages),
        stage_history=stages, final_stage_newton_delta=jnp.zeros_like(stages),
        stage_secant_applied=jnp.asarray(False), jacobian_out=carry.jacobian,
        cache_valid_out=carry.cache_valid, cache_dt_out=dt, cache_age_out=zero_int,
        real_lu_out=carry.real_lu, real_piv_out=carry.real_piv,
        complex_lu_out=carry.complex_lu, complex_piv_out=carry.complex_piv,
        theta_final=zero, newton_iter_count=zero_int,
    )
    rows = jnp.sin(jnp.arange(120, dtype=dtype).reshape((10, 12)) * 0.31)
    return kernel, physics, carry, primal, rows


def _assert_tree_close(actual, expected):
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(expected)
    for actual_leaf, expected_leaf in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
        assert np.all(np.isfinite(actual_leaf))
        np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=2e-12, atol=2e-13)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("stage_mode", ["shared", "shared_multi_rhs"])
def test_shared_stage_jacobian_matches_independent_pair_and_residual_ad(
    compiled, stage_mode
):
    kernel, physics, carry, primal, rows = _case()
    shared_physics = dataclasses.replace(
        physics, reverse_database_stage_jacobian_mode=stage_mode
    )

    def evaluate(t, y, dt, stages, rhs_rows):
        dynamic_carry = dataclasses.replace(carry, t=t, y=y)
        dynamic_primal = dataclasses.replace(primal, trial_dt=dt, stage_history=stages)
        independent_residuals = solvers._radau_solve_exact_stage_residual_transpose_batched(
            kernel, physics, dynamic_carry, dynamic_primal, None, rhs=rhs_rows,
        )
        independent_state = jax.vmap(lambda bar: solvers._radau_exact_stage_residual_input_pullback(
            kernel, physics, dynamic_carry, dynamic_primal, None, bar, compute_dt_bar=False,
        )[0])(independent_residuals)
        shared = solvers._radau_database_shared_stage_solve_and_state_pullback_batched(
            kernel, shared_physics, dynamic_carry, dynamic_primal, rhs=rhs_rows,
        )

        # Independent residual AD checks block layout, transpose, sign and y bar.
        def residual(y_value, stage_vector):
            stage_values = stage_vector.reshape((3, 4))
            states = y_value[None, :] + dt * (kernel.a @ stage_values)
            values = jax.vmap(physics.flat_rhs)(t + kernel.c * dt, states)
            return (stage_values - values).reshape((-1,))

        matrix = jax.jacfwd(lambda vector: residual(y, vector))(stages.reshape((-1,)))
        oracle_residuals = jax.vmap(lambda row: jnp.linalg.solve(matrix.T, -row))(rhs_rows)
        _, pullback_y = jax.vjp(lambda value: residual(value, stages.reshape((-1,))), y)
        oracle_state = jax.vmap(lambda bar: pullback_y(bar)[0])(oracle_residuals)
        return shared, (independent_residuals, independent_state), (oracle_residuals, oracle_state)

    run = jax.jit(evaluate) if compiled else evaluate
    for offset in (0.0, 0.11):
        stage_states = (
            carry.y[None, :]
            + offset
            + primal.trial_dt
            * (kernel.a @ (primal.stage_history - offset))
        )
        stage_times = carry.t + offset + kernel.c * primal.trial_dt
        stage_jacobians = jax.vmap(jax.jacfwd(physics.flat_rhs, argnums=1))(
            stage_times, stage_states
        )
        # The fixture is genuinely nonlinear and exercises distinct J_i; a
        # shared frozen Jacobian could otherwise pass this parity test.
        assert not np.allclose(stage_jacobians[0], stage_jacobians[-1])
        shared, independent, oracle = run(
            carry.t + offset, carry.y + offset, primal.trial_dt,
            primal.stage_history - offset, rows,
        )
        assert shared[0].shape == (10, 12)
        assert shared[1].shape == (10, 4)
        _assert_tree_close(shared, independent)
        _assert_tree_close(shared, oracle)


def test_shared_stage_jacobian_uses_one_explicit_multi_rhs_solve(monkeypatch):
    """The objective batch is columns of one solve, not mapped scalar solves."""
    kernel, physics, carry, primal, rows = _case()
    physics = dataclasses.replace(
        physics, reverse_database_stage_jacobian_mode="shared_multi_rhs"
    )
    original_solve = jnp.linalg.solve
    solve_shapes = []

    def tracked_solve(matrix, rhs):
        solve_shapes.append((matrix.shape, rhs.shape))
        return original_solve(matrix, rhs)

    monkeypatch.setattr(solvers.jnp.linalg, "solve", tracked_solve)
    residual_bars, state_bars = (
        solvers._radau_database_shared_stage_solve_and_state_pullback_batched(
            kernel, physics, carry, primal, rhs=rows,
        )
    )

    system_size = kernel.num_stages * kernel.state_dim
    assert solve_shapes == [
        ((system_size, system_size), (system_size, rows.shape[0]))
    ]
    assert residual_bars.shape == rows.shape
    assert state_bars.shape == (rows.shape[0], kernel.state_dim)
    assert np.all(np.isfinite(residual_bars))
    assert np.all(np.isfinite(state_bars))


@pytest.mark.parametrize("field, value", [
    ("reverse_database_stage_jacobian_mode", "unknown"),
    ("reverse_stage_adjoint_solve_mode", "structured"),
    ("reverse_stage_adjoint_solve_mode", "block_database_multi_rhs"),
    ("reverse_rhs_transpose_mode", "explicit_ntx_interpolated"),
    ("reverse_rhs_transpose_mode", "generic"),
    ("reverse_stage_cotangent_mode", "zero_rhs_state"),
    ("reverse_stage_adjoint_memory_mode", "stage_call_boundary"),
    ("reverse_rhs_pullback_mode", "fused_ntx"),
])
def test_shared_stage_jacobian_rejects_unsupported_contracts(field, value):
    _, physics, _, _, _ = _case()
    physics = dataclasses.replace(physics, reverse_database_stage_jacobian_mode="shared")
    physics = dataclasses.replace(physics, **{field: value})
    with pytest.raises(ValueError, match=field):
        solvers._radau_database_shared_stage_jacobian_enabled(physics, None)


def test_shared_stage_jacobian_rejects_lagged_rhs_and_independent_helper_call():
    kernel, physics, carry, primal, rows = _case()
    assert physics.reverse_database_stage_jacobian_mode == "independent"
    assert not solvers._radau_database_shared_stage_jacobian_enabled(physics, object())
    assert not solvers._radau_database_shared_stage_jacobian_enabled(SimpleNamespace(), None)
    shared_physics = dataclasses.replace(physics, reverse_database_stage_jacobian_mode="shared")
    with pytest.raises(ValueError, match="without a lagged response"):
        solvers._radau_database_shared_stage_jacobian_enabled(shared_physics, object())
    with pytest.raises(ValueError, match="requires.*shared"):
        solvers._radau_database_shared_stage_solve_and_state_pullback_batched(
            kernel, physics, carry, primal, rhs=rows,
        )


@pytest.mark.parametrize("shared_mode", [False, True], ids=["default-old-pair", "shared-bypasses-old-pair"])
def test_batched_database_support_core_dispatch_preserves_outputs(monkeypatch, shared_mode):
    kernel, physics, carry, primal, rows = _case()
    next_bars = solvers._RadauAcceptedStepReducedCotangent(
        y=rows[:, :4], lagged_response_cache=None,
        lagged_reference_y=jnp.zeros((10, 4), dtype=kernel.dtype),
    )
    support = {"database": jnp.asarray([0.2, 0.7]), "geometry": jnp.asarray([0.4, 0.1, -0.2])}
    core = solvers._execute_radau_accepted_step_next_reduced_cotangent_batched_bwd_with_support_from_primal_result_core

    def run(context):
        return core(
            kernel, context, None, "rebuild", carry, primal, next_bars, support,
            collect_database_geometry_record=True,
        )

    expected = run(physics)
    calls = []

    def forbidden(*args, **kwargs):
        raise AssertionError("Unselected stage Jacobian path was executed.")

    if shared_mode:
        helper = solvers._radau_database_shared_stage_solve_and_state_pullback_batched

        def tracked_helper(*args, **kwargs):
            calls.append("shared")
            return helper(*args, **kwargs)

        monkeypatch.setattr(solvers, "_radau_database_shared_stage_solve_and_state_pullback_batched", tracked_helper)
        monkeypatch.setattr(solvers, "_radau_solve_exact_stage_residual_transpose_batched", forbidden)
        monkeypatch.setattr(solvers, "_radau_exact_stage_residual_input_pullback", forbidden)
        physics = dataclasses.replace(physics, reverse_database_stage_jacobian_mode="shared")
    else:
        monkeypatch.setattr(solvers, "_radau_database_shared_stage_solve_and_state_pullback_batched", forbidden)

    actual = run(physics)
    _assert_tree_close(actual, expected)
    assert calls == (["shared"] if shared_mode else [])
    # The existing stage table/geometry transpose receives the same residuals.
    assert len(actual[1]) == 2
    assert actual[1][0].shape == (10, 2)
    assert actual[1][1].shape == (10, 3)
    assert actual[2] is not None
