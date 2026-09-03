import dataclasses
import types

import jax.numpy as jnp
import pytest

import NEOPAX._transport_solvers as transport_solvers

from NEOPAX._transport_solvers import (
    DiffraxSolver,
    LIMMWBaselineSolver,
    NewtonThetaMethodSolver,
    RADAUSolver,
    T3DOuterThetaMethodSolver,
    ThetaMethodSolver,
    _T3DOuterThetaSolverConfig,
    _apply_radau_lean_timestep_controller,
    _RadauStepState,
    build_time_solver,
)


def _base_solver_parameters(**overrides):
    params = {
        "t0": 0.0,
        "t_final": 1.0,
        "dt": 0.1,
        "transport_solver_backend": "theta_newton",
        "min_step": 1.0e-12,
        "max_step": 0.25,
        "max_steps": 100,
        "theta_implicit": 1.0,
        "nonlinear_solver_tol": 1.0e-8,
        "nonlinear_solver_maxiter": 12,
        "save_n": 4,
        "rtol": 1.0e-6,
        "atol": 1.0e-8,
    }
    params.update(overrides)
    return params


def test_limm_w_config_keeps_current_flux_and_jacobian_reuse_separate():
    """LIMM-W may reuse its matrix, but never its physical RHS anchor."""

    config = transport_solvers._LIMMWSolverConfig(
        order=5,
        jacobian_reuse_mode="global_state_drift_max",
    )

    assert config.order == 5
    assert config.jacobian_reuse_mode == "global_state_drift_max"
    assert not hasattr(config, "rhs_mode")


@pytest.mark.parametrize("order", [0, 6])
def test_limm_w_config_rejects_unsupported_order(order):
    with pytest.raises(ValueError, match="limm_w_order"):
        transport_solvers._LIMMWSolverConfig(order=order)


def test_limm_w_published_o16_configuration_exposes_fixed_w3_and_w5_lanes():
    assert transport_solvers._LIMMWSolverConfig(order=3, coefficient_family="o16_published").coefficient_family == "o16_published"
    assert transport_solvers._LIMMWSolverConfig(order=5, coefficient_family="o16_published").coefficient_family == "o16_published"
    with pytest.raises(ValueError, match="order 3 or 5"):
        transport_solvers._LIMMWSolverConfig(order=4, coefficient_family="o16_published")


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_limm_w_baseline_coefficients_satisfy_variable_step_order_conditions(order):
    """The private baseline is a valid W family before stability optimization."""

    dtype = jnp.float64
    current_dt = jnp.asarray(0.07, dtype=dtype)
    previous_dts = jnp.asarray([0.05, 0.09, 0.06, 0.08], dtype=dtype)
    beta, mu_past, gamma = transport_solvers._limm_w_baseline_coefficients(
        current_dt, previous_dts, order=order, dtype=dtype
    )
    c = jnp.concatenate(
        (
            jnp.zeros((1,), dtype=dtype),
            jnp.cumsum(previous_dts[: order - 1]) / current_dt,
        )
    )
    powers = jnp.arange(order, dtype=dtype)

    # Explicit Adams--Bashforth consistency conditions.
    assert jnp.allclose(
        ((-c)[None, :] ** powers[:, None]) @ beta,
        1.0 / (powers + 1.0),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    # W correction: sum_i mu_i c_i^m + gamma (-1)^m = 0.
    assert jnp.allclose(
        (c[None, :] ** powers[:, None]) @ mu_past
        + gamma * ((-jnp.ones_like(powers)) ** powers),
        jnp.zeros_like(powers),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_limm_w_baseline_trial_step_preserves_constant_rhs(order):
    """A response-only trial is exact for a constant nonlinear RHS sample."""

    dtype = jnp.float64
    h = jnp.asarray(0.1, dtype=dtype)
    y = jnp.asarray([1.5, -2.0], dtype=dtype)
    f = jnp.asarray([3.0, -4.0], dtype=dtype)
    state_history = jnp.broadcast_to(y, (order, y.size))
    rhs_history = jnp.broadcast_to(f, (order, y.size))
    y_new = transport_solvers._limm_w_baseline_trial_step(
        y,
        rhs_history,
        state_history,
        jnp.zeros((y.size, y.size), dtype=dtype),
        h,
        jnp.full((max(order - 1, 0),), h, dtype=dtype),
        order=order,
    )
    assert jnp.allclose(y_new, y + h * f, rtol=1.0e-12, atol=1.0e-12)


def test_limm_w_baseline_trial_step_uses_one_response_matrix_solve():
    """The kernel is an explicit RHS/history combination plus one LHS solve."""

    dtype = jnp.float64
    h = jnp.asarray(0.1, dtype=dtype)
    y = jnp.asarray([1.0], dtype=dtype)
    jacobian = jnp.asarray([[-5.0]], dtype=dtype)
    rhs_history = jnp.asarray([[-5.0], [-6.0]], dtype=dtype)
    state_history = jnp.asarray([[1.0], [1.2]], dtype=dtype)
    beta, mu, gamma = transport_solvers._limm_w_baseline_coefficients(
        h, jnp.asarray([h], dtype=dtype), order=2, dtype=dtype
    )
    expected = jnp.linalg.solve(
        jnp.eye(1, dtype=dtype) - h * gamma * jacobian,
        y
        + h * jnp.einsum("i,ij->j", beta, rhs_history)
        + h * (jacobian @ jnp.einsum("i,ij->j", mu, state_history)),
    )
    actual = transport_solvers._limm_w_baseline_trial_step(
        y,
        rhs_history,
        state_history,
        jacobian,
        h,
        jnp.asarray([h], dtype=dtype),
        order=2,
    )
    assert jnp.allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_limm_w_general_multistep_kernel_matches_provisional_coefficient_adapter():
    """The published alpha/beta/gamma form must preserve the old baseline."""

    dtype = jnp.float64
    order = 3
    h = jnp.asarray(0.08, dtype=dtype)
    previous_dts = jnp.asarray([0.07, 0.09], dtype=dtype)
    states = jnp.asarray([[1.0, -0.5], [0.9, -0.45], [0.8, -0.4]], dtype=dtype)
    rhs = jnp.asarray([[0.2, 0.3], [0.1, 0.25], [0.05, 0.2]], dtype=dtype)
    jacobian = jnp.asarray([[2.0, 0.1], [0.0, -1.5]], dtype=dtype)
    beta, gamma_past, gamma_new = transport_solvers._limm_w_baseline_coefficients(
        h, previous_dts, order=order, dtype=dtype
    )
    alpha = jnp.asarray([-1.0, 0.0, 0.0], dtype=dtype)
    generic = transport_solvers._limm_w_linear_multistep_trial_step(
        rhs_history=rhs,
        state_history=states,
        current_jacobian=jacobian,
        current_dt=h,
        alpha_past=alpha,
        beta_past=beta,
        gamma_past=gamma_past,
        gamma_new=gamma_new,
    )
    baseline = transport_solvers._limm_w_baseline_trial_step(
        states[0], rhs, states, jacobian, h, previous_dts, order=order
    )
    assert jnp.allclose(generic, baseline, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize("order", [1, 2, 3, 4, 5])
def test_limm_w_published_o16_coefficients_preserve_constant_rhs(order):
    """The vendored public variable-step tables satisfy the basic consistency law."""

    dtype = jnp.float64
    h = jnp.asarray(0.08, dtype=dtype)
    previous_dts = jnp.asarray([0.11, 0.05, 0.09, 0.06], dtype=dtype)
    alpha, beta, gamma_past, gamma_new = transport_solvers._limm_w_o16_coefficients(
        h, previous_dts, order=order, dtype=dtype
    )
    y = jnp.asarray([1.5, -2.0], dtype=dtype)
    f = jnp.asarray([3.0, -4.0], dtype=dtype)
    past_offsets = jnp.concatenate((jnp.zeros((1,), dtype=dtype), jnp.cumsum(previous_dts[: order - 1])))
    state_history = y[None, :] - past_offsets[:, None] * f[None, :]
    trial = transport_solvers._limm_w_linear_multistep_trial_step(
        rhs_history=jnp.broadcast_to(f, (order, f.size)),
        state_history=state_history,
        current_jacobian=jnp.zeros((y.size, y.size), dtype=dtype),
        current_dt=h,
        alpha_past=alpha,
        beta_past=beta,
        gamma_past=gamma_past,
        gamma_new=gamma_new,
    )
    assert alpha.shape == beta.shape == gamma_past.shape == (order,)
    assert jnp.isfinite(gamma_new)
    assert jnp.allclose(trial, y + h * f, rtol=1.0e-11, atol=1.0e-11)


def test_limm_w_history_advances_only_from_accepted_values():
    dtype = jnp.float64
    history = transport_solvers._limm_w_initial_history(
        jnp.asarray([1.0, 2.0], dtype=dtype),
        jnp.asarray([3.0, 4.0], dtype=dtype),
        max_order=3,
    )
    advanced = transport_solvers._limm_w_advance_history_on_accept(
        history,
        jnp.asarray([5.0, 6.0], dtype=dtype),
        jnp.asarray([7.0, 8.0], dtype=dtype),
        jnp.asarray(0.1, dtype=dtype),
    )

    assert jnp.array_equal(history.states[0], jnp.asarray([1.0, 2.0], dtype=dtype))
    assert jnp.array_equal(advanced.states[:2], jnp.asarray([[5.0, 6.0], [1.0, 2.0]], dtype=dtype))
    assert jnp.array_equal(advanced.rhs_values[:2], jnp.asarray([[7.0, 8.0], [3.0, 4.0]], dtype=dtype))
    assert float(advanced.accepted_dts[0]) == pytest.approx(0.1)
    assert int(advanced.valid_count) == 2


def test_limm_w_step_state_ramps_order_with_accepted_history():
    dtype = jnp.float64
    state = transport_solvers._limm_w_initial_step_state(
        0.0,
        jnp.asarray([1.0], dtype=dtype),
        jnp.asarray([-2.0], dtype=dtype),
        0.1,
        max_order=5,
    )
    assert int(transport_solvers._limm_w_available_order(state, 5)) == 1
    state = dataclasses.replace(
        state,
        history=transport_solvers._limm_w_advance_history_on_accept(
            state.history,
            jnp.asarray([0.8], dtype=dtype),
            jnp.asarray([-1.6], dtype=dtype),
            jnp.asarray(0.1, dtype=dtype),
        ),
    )
    assert int(transport_solvers._limm_w_available_order(state, 5)) == 1
    state = dataclasses.replace(
        state,
        history=transport_solvers._limm_w_advance_history_on_accept(
            state.history,
            jnp.asarray([0.7], dtype=dtype),
            jnp.asarray([-1.4], dtype=dtype),
            jnp.asarray(0.1, dtype=dtype),
        ),
    )
    assert int(transport_solvers._limm_w_available_order(state, 5)) == 2
    assert int(transport_solvers._limm_w_available_order(state, 1)) == 1


def test_limm_w_embedded_error_uses_only_current_response_and_history():
    dtype = jnp.float64
    h = jnp.asarray(0.1, dtype=dtype)
    y = jnp.asarray([1.0], dtype=dtype)
    jacobian = jnp.asarray([[-2.0]], dtype=dtype)
    states = jnp.asarray([[1.0], [1.1], [1.3]], dtype=dtype)
    rhs = jnp.asarray([[-2.0], [-2.2], [-2.6]], dtype=dtype)
    trial = transport_solvers._limm_w_baseline_trial_step(
        y, rhs, states, jacobian, h, jnp.asarray([h, h], dtype=dtype), order=2
    )
    error = transport_solvers._limm_w_embedded_trial_error(
        trial_y=trial,
        current_y=y,
        rhs_history=rhs,
        state_history=states,
        current_jacobian=jacobian,
        current_dt=h,
        previous_dts=jnp.asarray([h, h], dtype=dtype),
        order=2,
    )
    higher = transport_solvers._limm_w_baseline_trial_step(
        y, rhs, states, jacobian, h, jnp.asarray([h, h], dtype=dtype), order=3
    )
    assert jnp.allclose(error, trial - higher, rtol=1.0e-12, atol=1.0e-12)


def test_limm_w_fixed_pi_controller_shrinks_and_grows_dt_from_scaled_error():
    dtype = jnp.float64
    y = jnp.asarray([1.0, 2.0], dtype=dtype)
    err_norm = transport_solvers._limm_w_scaled_error_norm(
        jnp.asarray([0.1, 0.2], dtype=dtype), y, y, rtol=0.1, atol=0.0
    )
    assert float(err_norm) == pytest.approx(1.0)
    shrink = transport_solvers._limm_w_fixed_pi_controller(
        trial_dt=0.1, error_norm=100.0, previous_accepted_error=1.0,
        error_order=4, reject_last=False, min_step=1.0e-14, max_step=1.0,
        safety_factor=0.9, min_step_factor=0.2, max_step_factor=2.0,
    )
    grow = transport_solvers._limm_w_fixed_pi_controller(
        trial_dt=0.1, error_norm=1.0e-8, previous_accepted_error=1.0,
        error_order=4, reject_last=False, min_step=1.0e-14, max_step=1.0,
        safety_factor=0.9, min_step_factor=0.2, max_step_factor=2.0,
    )
    assert float(shrink.next_dt) < 0.1
    assert float(grow.next_dt) > 0.1


def test_limm_w_fixed_pi_retry_never_increases_step():
    retry = transport_solvers._limm_w_fixed_pi_controller(
        trial_dt=jnp.asarray(0.1), error_norm=jnp.asarray(4.0),
        previous_accepted_error=jnp.asarray(0.1), error_order=4,
        reject_last=jnp.asarray(True), min_step=1.0e-8, max_step=1.0,
        safety_factor=0.9, min_step_factor=0.2, max_step_factor=2.0,
    )
    assert not bool(retry.accepted)
    assert float(retry.next_dt) < 0.1
    assert bool(retry.reject_last)


def test_limm_w_attempt_commit_shifts_history_on_fixed_order_startup_accept():
    dtype = jnp.float64
    state = transport_solvers._limm_w_initial_step_state(
        0.0,
        jnp.asarray([1.0], dtype=dtype),
        jnp.asarray([-10.0], dtype=dtype),
        0.1,
        max_order=4,
    )

    accept_solver = transport_solvers._LIMMWSolverConfig(
        t0=0.0, t1=1.0, dt=0.1, order=3, rtol=1.0e3, atol=1.0e3
    )
    accepted_attempt = transport_solvers._limm_w_attempt_at_order(
        state,
        jnp.asarray([-10.0], dtype=dtype),
        jnp.asarray([[-10.0]], dtype=dtype),
        t_final=1.0,
        solver=accept_solver,
        order=1,
    )
    assert bool(accepted_attempt.accepted)
    accepted = transport_solvers._limm_w_commit_attempt(
        state, accepted_attempt, jnp.asarray([-5.0], dtype=dtype)
    )
    assert float(accepted.t) == pytest.approx(0.1)
    assert jnp.array_equal(accepted.history.states[0], accepted.y)
    assert jnp.array_equal(accepted.history.rhs_values[0], jnp.asarray([-5.0], dtype=dtype))
    assert int(accepted.status[2]) == 1


def test_limm_w_fixed_order_rejection_retains_every_accepted_history_row():
    """A rejected W3 retry may change h, but must never restart history."""

    dtype = jnp.float64
    state = transport_solvers._limm_w_initial_step_state(
        0.0, jnp.asarray([1.0], dtype=dtype), jnp.asarray([-1.0], dtype=dtype), 0.1, max_order=4
    )
    history = transport_solvers._LIMMWHistory(
        states=jnp.asarray([[1.0], [0.9], [0.8], [0.7]], dtype=dtype),
        rhs_values=jnp.asarray([[-1.0], [-0.8], [-0.6], [-0.4]], dtype=dtype),
        accepted_dts=jnp.asarray([0.1, 0.1, 0.1], dtype=dtype),
        valid_count=jnp.asarray(4, dtype=jnp.int32),
    )
    state = dataclasses.replace(state, history=history)
    solver = transport_solvers._LIMMWSolverConfig(
        t0=0.0, t1=1.0, dt=0.1, order=3, coefficient_family="o16_published", rtol=1.0e-14, atol=1.0e-14
    )
    attempt = transport_solvers._limm_w_attempt_at_order(
        state, history.rhs_values[0], jnp.zeros((1, 1), dtype=dtype), t_final=1.0, solver=solver, order=3
    )
    assert not bool(attempt.accepted)
    retried = transport_solvers._limm_w_commit_attempt(state, attempt, history.rhs_values[0])
    assert float(retried.t) == pytest.approx(float(state.t))
    assert jnp.array_equal(retried.history.states, state.history.states)
    assert jnp.array_equal(retried.history.rhs_values, state.history.rhs_values)
    assert jnp.array_equal(retried.history.accepted_dts, state.history.accepted_dts)
    assert int(retried.restart_count) == 0


def test_limm_w_full_rhs_adapter_keeps_sources_direct_and_response_only_in_jacobian():
    """The response accelerates dF/dy; F itself remains the full direct RHS."""

    dtype = jnp.float64
    state = transport_solvers._limm_w_initial_step_state(
        0.0,
        jnp.asarray([2.0], dtype=dtype),
        jnp.asarray([10.0], dtype=dtype),
        0.1,
        max_order=2,
    )
    # A structurally compatible cache is required for the compiled cond path.
    state = dataclasses.replace(
        state,
        reuse_state=dataclasses.replace(
            state.reuse_state,
            response_cache=jnp.asarray([2.0], dtype=dtype),
            response_available=jnp.asarray(True),
        ),
    )

    def full_rhs(_t, y):
        return y * y + 3.0 * y  # nonlinear flux-like term + a separate source

    def build_response(y):
        return y

    def response_rhs(_t, y, response):
        return response * response + 2.0 * response * (y - response) + 3.0 * y

    current_rhs, jacobian, reuse = transport_solvers._limm_w_prepare_full_rhs_and_jacobian(
        state,
        flat_rhs=full_rhs,
        build_lagged_response=build_response,
        flat_rhs_with_lagged_response=response_rhs,
        jacobian_reuse_mode="fresh",
    )
    assert jnp.allclose(current_rhs, jnp.asarray([10.0], dtype=dtype))
    assert jnp.allclose(jacobian, jnp.asarray([[7.0]], dtype=dtype))
    assert not bool(reuse.last_jacobian_reused)


def test_limm_w_anchor_build_uses_one_response_for_exact_full_rhs_and_jacobian():
    """A rebuilt accepted anchor must not duplicate its direct NTX call."""

    y0 = jnp.asarray([2.0])
    t0 = jnp.asarray(0.0)
    calls = {"build": 0, "direct": 0}

    def flat_rhs(_t, y):
        calls["direct"] += 1
        return y**2 + 3.0 * y

    def build_response(y):
        calls["build"] += 1
        return y**2

    def response_rhs(_t, y, response):
        # Linearized NTX flux plus the direct non-NTX source.
        return response + 2.0 * jnp.sqrt(response) * (y - jnp.sqrt(response)) + 3.0 * y

    rhs0, reuse0 = transport_solvers._limm_w_initial_anchor_full_rhs_and_jacobian(
        y0,
        t0=t0,
        flat_rhs=flat_rhs,
        build_lagged_response=build_response,
        flat_rhs_with_lagged_response=response_rhs,
    )

    assert calls == {"build": 1, "direct": 0}
    assert jnp.allclose(rhs0, y0**2 + 3.0 * y0)
    assert jnp.allclose(reuse0.jacobian, jnp.asarray([[7.0]]))

    y1 = jnp.asarray([3.0])
    rhs1, reuse1 = transport_solvers._limm_w_prepare_next_accepted_anchor(
        y1,
        accepted_t=jnp.asarray(0.1),
        reuse_state=reuse0,
        flat_rhs=flat_rhs,
        build_lagged_response=build_response,
        flat_rhs_with_lagged_response=response_rhs,
        jacobian_reuse_mode="fresh",
    )
    # ``lax.cond`` traces both branches in eager mode, so Python counters are
    # not an execution-count instrument.  The solver invariant is instead
    # that the selected rebuilt branch returns the exact anchor F and J.
    assert calls["build"] >= 2
    assert jnp.allclose(rhs1, y1**2 + 3.0 * y1)
    assert jnp.allclose(reuse1.jacobian, jnp.asarray([[9.0]]))


def test_limm_w_anchor_global_jacobian_reuse_keeps_rhs_direct():
    """Reusing J across anchors must never reuse the old response value F."""

    y0 = jnp.asarray([2.0])
    calls = {"build": 0, "direct": 0}

    def flat_rhs(_t, y):
        calls["direct"] += 1
        return y**2 + 3.0 * y

    def build_response(y):
        calls["build"] += 1
        return y**2

    def response_rhs(_t, y, response):
        return response + 2.0 * jnp.sqrt(response) * (y - jnp.sqrt(response)) + 3.0 * y

    _, reuse0 = transport_solvers._limm_w_initial_anchor_full_rhs_and_jacobian(
        y0,
        t0=jnp.asarray(0.0),
        flat_rhs=flat_rhs,
        build_lagged_response=build_response,
        flat_rhs_with_lagged_response=response_rhs,
    )
    y1 = jnp.asarray([2.001])
    rhs1, reuse1 = transport_solvers._limm_w_prepare_next_accepted_anchor(
        y1,
        accepted_t=jnp.asarray(0.1),
        reuse_state=reuse0,
        flat_rhs=flat_rhs,
        build_lagged_response=build_response,
        flat_rhs_with_lagged_response=response_rhs,
        jacobian_reuse_mode="global_state_drift_max",
        jacobian_reuse_rtol=1.0e-2,
        jacobian_reuse_atol=1.0e-12,
    )
    assert calls["direct"] >= 1
    assert bool(reuse1.last_jacobian_reused)
    assert jnp.allclose(rhs1, y1**2 + 3.0 * y1)


def test_limm_w_baseline_forward_harness_advances_complete_rhs_without_response_hook():
    """The private forward lane is usable before backend/TOML exposure."""

    solver = LIMMWBaselineSolver(
        t0=0.0,
        t1=0.2,
        dt=0.05,
        order=3,
        rtol=1.0e-8,
        atol=1.0e-10,
        max_steps=20,
        save_n=3,
    )
    out = solver.solve(jnp.asarray([1.0]), lambda _t, y: jnp.ones_like(y))

    assert not bool(out["failed"])
    assert float(out["final_time"]) == pytest.approx(0.2)
    assert jnp.allclose(out["final_state"], jnp.asarray([1.2]), rtol=1.0e-12, atol=1.0e-12)
    assert int(out["n_steps"]) >= 1


def test_limm_w_published_o16_forward_harness_advances_complete_rhs_without_response_hook():
    """The production candidate uses the same accepted-anchor physics path."""

    solver = LIMMWBaselineSolver(
        t0=0.0,
        t1=0.2,
        dt=0.05,
        order=3,
        coefficient_family="o16_published",
        rtol=1.0e-8,
        atol=1.0e-10,
        max_steps=20,
        save_n=3,
    )
    out = solver.solve(jnp.asarray([1.0]), lambda _t, y: jnp.ones_like(y))

    assert not bool(out["failed"])
    assert float(out["final_time"]) == pytest.approx(0.2)
    assert jnp.allclose(out["final_state"], jnp.asarray([1.2]), rtol=1.0e-12, atol=1.0e-12)


def test_limm_w_published_o16_compiled_loop_controls_stiff_linear_rhs():
    """Regression for the compiled adaptive path on y' = -100 y."""

    solver = LIMMWBaselineSolver(
        t0=0.0,
        t1=0.1,
        dt=1.0e-3,
        max_step=2.0e-2,
        order=3,
        coefficient_family="o16_published",
        rtol=1.0e-6,
        atol=1.0e-10,
        max_steps=2000,
    )
    out = solver.solve(jnp.asarray([1.0]), lambda _t, y: -100.0 * y)

    assert not bool(out["failed"])
    assert float(out["final_time"]) == pytest.approx(0.1)
    assert jnp.allclose(out["final_state"], jnp.asarray([jnp.exp(-10.0)]), rtol=2.0e-3, atol=2.0e-7)


def test_limm_w5_uses_fixed_order_history_and_lower_embedded_companion():
    """W5 must reach its fixed production pair without a p=6 table/NTX call."""

    solver = LIMMWBaselineSolver(
        t0=0.0,
        t1=0.3,
        dt=0.02,
        order=5,
        coefficient_family="o16_published",
        rtol=1.0e-6,
        atol=1.0e-10,
        max_steps=200,
    )
    out = solver.solve(jnp.asarray([1.0]), lambda _t, y: jnp.ones_like(y))

    assert not bool(out["failed"])
    assert float(out["final_time"]) == pytest.approx(0.3)
    assert jnp.allclose(out["final_state"], jnp.asarray([1.3]), rtol=1.0e-11, atol=1.0e-11)


def test_limm_w_history_predictor_reproduces_linear_trajectory():
    dtype = jnp.float64
    # y(t) = 3 - 2 t; current state is at t=0 and older accepted states are
    # at -0.1 and -0.3, exactly matching the variable-step history convention.
    history = jnp.asarray([[3.0], [3.2], [3.6]], dtype=dtype)
    predicted = transport_solvers._limm_w_history_predictor(
        history, jnp.asarray([0.1, 0.2], dtype=dtype), jnp.asarray(0.05, dtype=dtype), order=3
    )
    assert jnp.allclose(predicted, jnp.asarray([2.9], dtype=dtype), rtol=1.0e-12, atol=1.0e-12)



def test_lagged_response_global_state_drift_max_uses_componentwise_max_norm():
    """A single out-of-tolerance component must invalidate max-norm reuse."""

    reference = jnp.zeros(4)
    current = jnp.asarray([1.2, 0.0, 0.0, 0.0])
    rms = transport_solvers._lagged_response_global_reuse_metric(
        current, reference, atol=1.0, rtol=0.0, norm="rms"
    )
    max_norm = transport_solvers._lagged_response_global_reuse_metric(
        current, reference, atol=1.0, rtol=0.0, norm="max"
    )

    assert float(rms) < 1.0
    assert float(max_norm) == pytest.approx(1.2)
    assert transport_solvers._lagged_response_reuse_uses_global_drift(
        "global_state_drift_max"
    )
    assert transport_solvers._lagged_response_drift_norm("global_state_drift_max") == "max"
    assert float(
        transport_solvers._lagged_response_global_reuse_metric(
            reference, reference, atol=0.0, rtol=0.0, norm="max"
        )
    ) == 0.0
    assert bool(
        jnp.isinf(
            transport_solvers._lagged_response_global_reuse_metric(
                current, reference, atol=0.0, rtol=0.0, norm="max"
            )
        )
    )


def test_theta_global_max_rebuilds_at_new_accepted_step_start():
    """A tight global-max threshold cannot accidentally retain the old anchor."""

    state_dim = 1
    dtype = jnp.float64
    step_state = transport_solvers._theta_initial_step_state(
        jnp.asarray(0.1, dtype=dtype),
        jnp.asarray([1.0], dtype=dtype),
        jnp.asarray(0.01, dtype=dtype),
        state_dim,
        dtype,
        lagged_response_cache=jnp.asarray([0.0], dtype=dtype),
        lagged_response_valid=True,
    )
    step_state = dataclasses.replace(
        step_state,
        reuse_state=dataclasses.replace(
            step_state.reuse_state,
            lagged_reference_y=jnp.asarray([0.0], dtype=dtype),
        ),
    )

    response, reference, reused = transport_solvers._theta_prepare_lagged_response(
        step_state,
        use_transport_lagged_response=True,
        lagged_response_reuse_mode="global_state_drift_max",
        lagged_response_reuse_rtol=1.0e-7,
        lagged_response_reuse_atol=1.0e-8,
        unpack_flat=lambda value: value,
        project_flat=None,
        build_lagged_response=lambda value: 10.0 * value,
    )

    assert not bool(reused)
    assert jnp.array_equal(reference, jnp.asarray([1.0], dtype=dtype))
    assert jnp.array_equal(response, jnp.asarray([10.0], dtype=dtype))


def test_theta_preserves_actual_anchor_after_global_response_reuse():
    """The cache anchor must not drift forward when the response is reused."""

    dtype = jnp.float32
    identity = jnp.eye(1, dtype=dtype)
    step_state = transport_solvers._theta_initial_step_state(
        jnp.asarray(0.1, dtype=dtype),
        jnp.asarray([0.05], dtype=dtype),
        jnp.asarray(0.01, dtype=dtype),
        1,
        dtype,
        lagged_response_cache=jnp.asarray([0.0], dtype=dtype),
        lagged_response_valid=True,
    )
    step_state = dataclasses.replace(
        step_state,
        reuse_state=dataclasses.replace(
            step_state.reuse_state,
            lagged_reference_y=jnp.asarray([0.0], dtype=dtype),
        ),
    )
    response, reference, reused = transport_solvers._theta_prepare_lagged_response(
        step_state,
        use_transport_lagged_response=True,
        lagged_response_reuse_mode="global_state_drift_max",
        lagged_response_reuse_rtol=0.0,
        lagged_response_reuse_atol=0.1,
        unpack_flat=lambda value: value,
        project_flat=None,
        build_lagged_response=lambda value: value,
    )
    context = transport_solvers._ThetaAttemptContext(
        t=jnp.asarray(0.1, dtype=dtype),
        y=step_state.y,
        trial_dt=jnp.asarray(0.01, dtype=dtype),
        t_new=jnp.asarray(0.11, dtype=dtype),
        f_old=step_state.y * step_state.y,
        lagged_response=response,
        lagged_reference_y=reference,
        lagged_response_reused=reused,
        reuse_state=step_state.reuse_state,
    )
    result = transport_solvers._theta_newton_accepted_step_attempt(
        context,
        predictor_mode="euler",
        n_linearized_solves=1,
        theta=jnp.asarray(1.0, dtype=dtype),
        one=jnp.asarray(1.0, dtype=dtype),
        identity_n=identity,
        flat_rhs=lambda _t, y: y * y,
        flat_rhs_with_lagged_response=lambda _t, y, response: response * response + 2.0 * response * (y - response),
        use_lagged_linear_response=False,
        use_transport_lagged_response=True,
        lagged_response_reuse_mode="global_state_drift_max",
        jacobian_reuse_rtol=jnp.asarray(0.1, dtype=dtype),
        max_jacobian_age=jnp.asarray(8, dtype=jnp.int32),
        delta_reduction_factor=jnp.asarray(0.5, dtype=dtype),
        tau_min=jnp.asarray(0.01, dtype=dtype),
        project_flat=None,
        dtype=dtype,
        tol=jnp.asarray(1.0e-6, dtype=dtype),
        maxiter=jnp.asarray(4, dtype=jnp.int32),
        debug_newton_trace=False,
    )

    assert bool(reused)
    assert jnp.array_equal(result.lagged_reference_y_out, jnp.asarray([0.0], dtype=dtype))


def test_build_time_solver_theta_newton_backend():
    pytest.importorskip("diffrax")
    solver = build_time_solver(_base_solver_parameters(transport_solver_backend="theta_newton"))
    assert isinstance(solver, NewtonThetaMethodSolver)
    assert float(solver.t0) == 0.0
    assert float(solver.t1) == 1.0
    assert solver.rhs_mode == "black_box"


def test_build_time_solver_t3d_outer_theta_backend():
    pytest.importorskip("diffrax")
    solver = build_time_solver(
        _base_solver_parameters(
            transport_solver_backend="theta_t3d_outer",
            theta_rhs_mode="lagged_transport_response",
        )
    )
    assert isinstance(solver, T3DOuterThetaMethodSolver)
    assert solver.rhs_mode == "lagged_transport_response"


def test_build_time_solver_limm_w_forward_backend_is_explicitly_forward_only():
    solver = build_time_solver(
        _base_solver_parameters(
            transport_solver_backend="limm_w_forward",
            limm_w_order=3,
            limm_w_coefficient_family="o16_published",
            limm_w_jacobian_reuse_mode="retry_only",
        )
    )
    assert isinstance(solver, LIMMWBaselineSolver)
    assert solver.coefficient_family == "o16_published"
    assert solver.jacobian_reuse_mode == "retry_only"


def test_t3d_outer_theta_solver_rebuilds_lagged_response_before_accepting_time():
    class QuadraticLaggedField:
        def __init__(self):
            self.anchors = []

        def __call__(self, _t, y):
            return y * y

        def build_lagged_response(self, y):
            self.anchors.append(float(y[0]))
            return y

        def evaluate_with_lagged_response(self, _t, y, *, lagged_response):
            return lagged_response * lagged_response + 2.0 * lagged_response * (y - lagged_response)

    field = QuadraticLaggedField()
    solver = T3DOuterThetaMethodSolver(
        t0=0.0,
        t1=0.1,
        dt=0.1,
        theta_implicit=1.0,
        outer_maxiter=2,
        outer_rms_threshold=1.0e-8,
        outer_rms_tolerance=1.0e-2,
        max_steps=4,
    )
    out = solver.solve(jnp.asarray([1.0]), field.__call__)

    assert field.anchors == pytest.approx([1.0, 1.125])
    assert int(out["n_steps"]) == 1
    assert float(out["final_time"]) == pytest.approx(0.1)
    assert jnp.asarray(out["ts"]).tolist() == pytest.approx([0.0, 0.1])
    assert jnp.asarray(out["accepted_mask"]).tolist() == [True, True]
    assert int(out["t3d_outer_iterations"]) == 2


def test_t3d_outer_theta_config_has_t3d_outer_iteration_contract():
    config = _T3DOuterThetaSolverConfig(
        t0=0.0,
        t1=1.0,
        dt=0.1,
        outer_maxiter=4,
        outer_rms_threshold=2.0e-2,
        outer_rms_tolerance=1.0e-1,
        dt_adjust=2.0,
    )
    assert config.rhs_mode == "lagged_transport_response"
    assert config.outer_maxiter == 4
    assert config.outer_rms_threshold == pytest.approx(2.0e-2)
    assert config.outer_rms_tolerance == pytest.approx(1.0e-1)
    assert config.dt_adjust == pytest.approx(2.0)
    assert config.dt_increase_threshold == pytest.approx(5.0e-3)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"outer_maxiter": 0}, "at least one"),
        ({"outer_rms_threshold": 0.2, "outer_rms_tolerance": 0.1}, "greater than or equal"),
        ({"dt_adjust": 1.0}, "greater than one"),
    ],
)
def test_t3d_outer_theta_config_rejects_invalid_outer_controls(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _T3DOuterThetaSolverConfig(t0=0.0, t1=1.0, dt=0.1, **kwargs)


def test_t3d_outer_theta_rebuilds_response_at_endpoint_iterates():
    anchors = []

    def build_response(anchor):
        anchors.append(float(anchor[0]))
        return anchor

    def direct_rhs(_t, y):
        return y * y

    def lagged_rhs(_t, y, response):
        return response * response + 2.0 * response * (y - response)

    result = transport_solvers._t3d_outer_theta_fixed_target(
        y_start=jnp.asarray([1.0]),
        t_start=jnp.asarray(0.0),
        dt=jnp.asarray(0.1),
        theta=jnp.asarray(1.0),
        flat_rhs=direct_rhs,
        flat_rhs_with_lagged_response=lagged_rhs,
        build_lagged_response_at_flat=build_response,
        rms_fn=lambda trial, anchor: jnp.linalg.norm(trial - anchor),
        outer_maxiter=2,
        outer_rms_threshold=1.0e-8,
        outer_rms_tolerance=1.0e-2,
    )

    # The first response is built at y_n.  The second is built at the first
    # candidate endpoint, while the theta left endpoint remains y_n.
    assert anchors == pytest.approx([1.0, 1.125])
    assert result.outer_iterations == 2
    assert not result.converged
    assert result.accepted
    assert not result.failed
    assert float(result.trial_y[0]) == pytest.approx(1.127016129, rel=1.0e-8)


def test_t3d_outer_theta_reports_failure_without_advancing_step_start():
    y_start = jnp.asarray([1.0])
    result = transport_solvers._t3d_outer_theta_fixed_target(
        y_start=y_start,
        t_start=jnp.asarray(0.0),
        dt=jnp.asarray(0.1),
        theta=jnp.asarray(1.0),
        flat_rhs=lambda _t, y: y * y,
        flat_rhs_with_lagged_response=lambda _t, y, response: response * response + 2.0 * response * (y - response),
        build_lagged_response_at_flat=lambda anchor: anchor,
        rms_fn=lambda trial, anchor: jnp.linalg.norm(trial - anchor),
        outer_maxiter=1,
        outer_rms_threshold=1.0e-12,
        outer_rms_tolerance=1.0e-12,
    )

    assert result.failed
    assert not result.accepted
    assert jnp.array_equal(y_start, jnp.asarray([1.0]))


def test_build_time_solver_theta_backend_accepts_shared_lagged_rhs_mode():
    pytest.importorskip("diffrax")
    solver = build_time_solver(
        _base_solver_parameters(
            transport_solver_backend="theta",
            rhs_mode="lagged_linear_state",
        )
    )
    assert isinstance(solver, ThetaMethodSolver)
    assert solver.rhs_mode == "lagged_linear_state"


def test_build_time_solver_theta_newton_backend_accepts_shared_lagged_rhs_mode():
    pytest.importorskip("diffrax")
    solver = build_time_solver(
        _base_solver_parameters(
            transport_solver_backend="theta_newton",
            rhs_mode="lagged_linear_state",
        )
    )
    assert isinstance(solver, NewtonThetaMethodSolver)
    assert solver.rhs_mode == "lagged_linear_state"


def test_build_time_solver_radau_backend():
    pytest.importorskip("diffrax")
    solver = build_time_solver(_base_solver_parameters(transport_solver_backend="radau"))
    assert isinstance(solver, RADAUSolver)
    assert float(solver.t0) == 0.0
    assert float(solver.t1) == 1.0
    assert solver.rhs_mode == "black_box"


@pytest.mark.parametrize("backend", ["radau", "theta_newton"])
def test_build_time_solver_accepts_global_state_drift_max(backend):
    pytest.importorskip("diffrax")
    reuse_key = (
        "lagged_response_reuse_mode"
        if backend == "radau"
        else "theta_lagged_response_reuse_mode"
    )
    solver = build_time_solver(
        _base_solver_parameters(
            transport_solver_backend=backend,
            **{reuse_key: "global_state_drift_max"},
        )
    )
    assert solver.lagged_response_reuse_mode == "global_state_drift_max"


def test_build_time_solver_radau_accepts_lagged_rhs_mode():
    pytest.importorskip("diffrax")
    solver = build_time_solver(
        _base_solver_parameters(
            transport_solver_backend="radau",
            radau_rhs_mode="lagged_linear_state",
        )
    )
    assert isinstance(solver, RADAUSolver)
    assert solver.rhs_mode == "lagged_linear_state"


def test_build_time_solver_radau_accepts_shared_lagged_rhs_mode():
    pytest.importorskip("diffrax")
    solver = build_time_solver(
        _base_solver_parameters(
            transport_solver_backend="radau",
            rhs_mode="lagged_linear_state",
        )
    )
    assert isinstance(solver, RADAUSolver)
    assert solver.rhs_mode == "lagged_linear_state"


def test_build_time_solver_radau_accepts_endpoint_defect_correction():
    solver = build_time_solver(
        _base_solver_parameters(
            transport_solver_backend="radau",
            radau_rhs_mode="lagged_transport_response",
            radau_lagged_response_correction_mode="endpoint_defect",
        )
    )
    assert isinstance(solver, RADAUSolver)
    assert solver.lagged_response_correction_mode == "endpoint_defect"


def test_radau_endpoint_defect_correction_requires_transport_lagged_response():
    with pytest.raises(ValueError, match="requires radau_rhs_mode='lagged_transport_response'"):
        RADAUSolver(lagged_response_correction_mode="endpoint_defect")


def test_radau_endpoint_defect_correction_runs_on_nonlinear_lagged_rhs():
    class QuadraticLaggedField:
        def __call__(self, _t, y):
            return y * y

        def build_lagged_response(self, y):
            return y

        def evaluate_with_lagged_response(self, _t, y, *, lagged_response):
            return lagged_response * lagged_response + 2.0 * lagged_response * (y - lagged_response)

    solver = RADAUSolver(
        t0=0.0,
        t1=0.05,
        dt=0.01,
        rtol=1.0e-5,
        atol=1.0e-8,
        rhs_mode="lagged_transport_response",
        lagged_response_correction_mode="endpoint_defect",
        maxiter=8,
        max_steps=32,
    )
    out = solver.solve(jnp.asarray([0.1]), QuadraticLaggedField().__call__)
    assert int(out["n_steps"]) > 0
    assert jnp.all(jnp.isfinite(out["final_state"]))


def test_build_time_solver_legacy_integrator_fallback():
    pytest.importorskip("diffrax")
    solver = build_time_solver(_base_solver_parameters(transport_solver_backend="diffrax_kvaerno5"))
    assert isinstance(solver, DiffraxSolver)
    assert float(solver.t0) == 0.0
    assert float(solver.t1) == 1.0


def test_build_time_solver_accepts_solver_instance_override():
    pytest.importorskip("diffrax")
    override = RADAUSolver(t0=0.0, t1=2.0, dt=0.2)
    solver = build_time_solver(_base_solver_parameters(), solver_override=override)
    assert solver is override


def test_theta_newton_solver_runs_scalar_decay_problem():
    solver = NewtonThetaMethodSolver(
        t0=0.0,
        t1=0.5,
        dt=0.05,
        theta_implicit=1.0,
        tol=1.0e-9,
        maxiter=20,
        max_steps=128,
        save_n=4,
    )

    def vector_field(t, y):
        del t
        return -2.0 * y

    out = solver.solve(jnp.array([1.0]), vector_field)
    final_state = out["final_state"]
    assert jnp.all(jnp.isfinite(final_state))
    assert final_state.shape == (1,)
    assert float(final_state[0]) < 1.0


def test_radau_transport_endpoint_newton_tolerance_runs_scalar_decay_problem():
    """The endpoint metric is an in-loop Newton criterion, not a fallback."""
    solver = RADAUSolver(
        t0=0.0,
        t1=0.1,
        dt=0.01,
        rtol=1.0e-6,
        atol=1.0e-8,
        tol=1.0e-7,
        maxiter=12,
        num_stages=3,
        newton_tol_mode="transport_endpoint",
        newton_fnewt_mode="hairer",
        newton_transport_endpoint_tol=0.1,
        max_steps=100,
        save_n=3,
    )

    def vector_field(t, y):
        del t
        return -2.0 * y

    out = solver.solve(jnp.array([1.0]), vector_field)
    assert not bool(out["failed"])
    assert float(out["final_state"][0]) == pytest.approx(float(jnp.exp(-0.2)), rel=2.0e-5)


def _radau_controller_arguments(*, newton_iter_count, theta_final=0.0, controller_mode="current"):
    dtype = jnp.float64
    y = jnp.zeros((2,), dtype=dtype)
    dt = jnp.asarray(1.0e-6, dtype=dtype)
    zero_int = jnp.asarray(0, dtype=jnp.int32)
    step_state = _RadauStepState(
        t=jnp.asarray(0.0, dtype=dtype),
        y=y,
        dt=dt,
        status=jnp.asarray([0, 0, 0], dtype=jnp.int32),
        prev_error=jnp.asarray(1.0, dtype=dtype),
        prev_stages=jnp.zeros((3, 2), dtype=dtype),
        prev_dt=dt,
        recent_reject_count=zero_int,
        regrowth_cooldown=zero_int,
        easy_growth_streak=zero_int,
        lagged_response_cache=None,
        lagged_response_valid=jnp.asarray(False),
        lagged_reference_y=y,
        jacobian=jnp.zeros((2, 2), dtype=dtype),
        cache_valid=jnp.asarray(False),
        cache_dt=dt,
        cache_age=zero_int,
        real_lu=jnp.zeros((2, 2), dtype=dtype),
        real_piv=jnp.zeros((2,), dtype=jnp.int32),
        complex_lu=jnp.zeros((2, 2), dtype=dtype),
        complex_piv=jnp.zeros((2,), dtype=jnp.int32),
        prev_theta_final=jnp.asarray(0.0, dtype=dtype),
        prev_newton_iter_count=zero_int,
    )
    return {
        "step_state": step_state,
        "trial_dt": dt,
        "trial_y": y,
        # Error norms small enough that the raw controller factor saturates at max_step_factor,
        # so the returned growth is whichever difficulty cap applies and nothing else.
        "err_norm": jnp.asarray(1.0e-8, dtype=dtype),
        "density_err_norm": jnp.asarray(1.0e-8, dtype=dtype),
        "pressure_err_norm": jnp.asarray(1.0e-8, dtype=dtype),
        "er_err_norm": jnp.asarray(1.0e-8, dtype=dtype),
        "converged": jnp.asarray(True),
        "stage_history": jnp.zeros((3, 2), dtype=dtype),
        "jacobian_out": jnp.zeros((2, 2), dtype=dtype),
        "cache_valid_out": jnp.asarray(False),
        "cache_dt_out": dt,
        "cache_age_out": zero_int,
        "real_lu_out": jnp.zeros((2, 2), dtype=dtype),
        "real_piv_out": jnp.zeros((2,), dtype=jnp.int32),
        "complex_lu_out": jnp.zeros((2, 2), dtype=dtype),
        "complex_piv_out": jnp.zeros((2,), dtype=jnp.int32),
        "newton_shrink": jnp.asarray(0.5, dtype=dtype),
        "diverged_final": jnp.asarray(False),
        "nonfinite_stage_state": jnp.asarray(False),
        "nonfinite_stage_residual": jnp.asarray(False),
        "finite_f0": jnp.asarray(True),
        "finite_z0": jnp.asarray(True),
        "finite_initial_residual": jnp.asarray(True),
        "newton_iter_count": jnp.asarray(newton_iter_count, dtype=jnp.int32),
        "final_residual_norm": jnp.asarray(1.0e-10, dtype=dtype),
        "final_delta_norm": jnp.asarray(1.0e-10, dtype=dtype),
        "theta_final": jnp.asarray(theta_final, dtype=dtype),
        "slow_contraction": jnp.asarray(False),
        "residual_blowup": jnp.asarray(False),
        "newton_nonfinite": jnp.asarray(False),
        "stagnation_accepted": jnp.asarray(False),
        "stagnation_defect_norm": jnp.asarray(0.0, dtype=dtype),
        "stagnation_growth_cap": jnp.asarray(1.25, dtype=dtype),
        "lagged_reused": jnp.asarray(False),
        "jacobian_reused": jnp.asarray(False),
        "fail_code": zero_int,
        "n_accepted": zero_int,
        "dtype": dtype,
        "dt_min": jnp.asarray(1.0e-12, dtype=dtype),
        "dt_max": jnp.asarray(1.0e-2, dtype=dtype),
        "safety_factor": jnp.asarray(0.9, dtype=dtype),
        "controller_alpha": jnp.asarray(0.175, dtype=dtype),
        "min_step_factor": jnp.asarray(0.1, dtype=dtype),
        "max_step_factor": jnp.asarray(5.0, dtype=dtype),
        "controller_mode": controller_mode,
        "use_transport_lagged_response": False,
        "lagged_response_reuse_mode": "current",
        "lagged_response_reuse_rtol": 1.0e-6,
        "lagged_response_reuse_atol": 1.0e-8,
        "project_flat": None,
    }


# The three controller modes that share the difficulty ladder; every hairer_* mode is exempt from it.
@pytest.mark.parametrize("controller_mode", ["current", "current_legacy", "gustafsson"])
def test_radau_controller_regrows_dt_after_a_difficult_accepted_step(controller_mode):
    _, info = _apply_radau_lean_timestep_controller(
        **_radau_controller_arguments(newton_iter_count=6, controller_mode=controller_mode)
    )
    assert float(info.growth) == pytest.approx(1.25)
    # Compare the ratio rather than next_dt itself so the assertion stays relative at any step size.
    assert float(info.next_dt / info.dt) == pytest.approx(1.25)


def test_radau_controller_holds_dt_after_a_very_difficult_accepted_step():
    _, info = _apply_radau_lean_timestep_controller(**_radau_controller_arguments(newton_iter_count=8))
    assert float(info.growth) == pytest.approx(1.0)


def test_radau_controller_caps_only_an_endpoint_correction_plateau_acceptance():
    args = _radau_controller_arguments(
        newton_iter_count=3,
        controller_mode="hairer_lean_transport_discounted",
    )
    args["stagnation_accepted"] = jnp.asarray(True)
    args["stagnation_defect_norm"] = jnp.asarray(0.4, dtype=jnp.float64)
    args["stagnation_growth_cap"] = jnp.asarray(1.1, dtype=jnp.float64)
    _, info = _apply_radau_lean_timestep_controller(**args)
    assert bool(info.accepted)
    assert bool(info.stagnation_accepted)
    assert float(info.stagnation_defect_norm) == pytest.approx(0.4)
    assert float(info.growth) == pytest.approx(1.1)


def test_radau_discounted_newton_bracket_avoids_immediate_regrowth_to_failed_dt():
    """A successful small retry must probe below, not jump past, a Newton failure."""

    args = _radau_controller_arguments(
        newton_iter_count=3,
        controller_mode="hairer_lean_transport_discounted_newton_bracket",
    )
    failed_dt = jnp.asarray(5.0e-6, dtype=jnp.float64)
    retry_dt = jnp.asarray(1.0e-6, dtype=jnp.float64)
    args["step_state"] = dataclasses.replace(
        args["step_state"],
        newton_reject_dt_upper=failed_dt,
    )
    args["trial_dt"] = retry_dt

    next_state, info = _apply_radau_lean_timestep_controller(**args)

    assert bool(info.accepted)
    # The uncorrected Hairer-lean controller would choose the 5x proposal.
    assert float(info.next_dt) == pytest.approx(0.9 * float(failed_dt))
    assert float(info.growth) == pytest.approx(0.9 * float(failed_dt) / float(retry_dt))
    assert float(info.next_dt) < float(failed_dt)
    assert float(next_state.newton_reject_dt_upper) == pytest.approx(float(failed_dt))


def test_radau_discounted_newton_bracket_records_only_newton_rejections():
    """An LTE rejection must not be mistaken for a stage-convergence bound."""

    args = _radau_controller_arguments(
        newton_iter_count=3,
        controller_mode="hairer_lean_transport_discounted_newton_bracket",
    )
    args["converged"] = jnp.asarray(False)
    args["err_norm"] = jnp.asarray(2.0, dtype=jnp.float64)
    failed_state, _ = _apply_radau_lean_timestep_controller(**args)
    assert float(failed_state.newton_reject_dt_upper) == pytest.approx(1.0e-6)

    args = _radau_controller_arguments(
        newton_iter_count=3,
        controller_mode="hairer_lean_transport_discounted_newton_bracket",
    )
    args["converged"] = jnp.asarray(True)
    args["err_norm"] = jnp.asarray(2.0, dtype=jnp.float64)
    lte_failed_state, _ = _apply_radau_lean_timestep_controller(**args)
    assert float(lte_failed_state.newton_reject_dt_upper) == pytest.approx(0.0)


def test_radau_discounted_newton_bracket_mode_is_an_explicit_opt_in():
    solver = RADAUSolver(
        t0=0.0,
        t1=1.0,
        dt=1.0e-3,
        controller_mode="discounted_newton_bracket",
    )
    assert solver.controller_mode == "hairer_lean_transport_discounted_newton_bracket"


def test_radau_discounted_newton_bracket_runs_through_jitted_adaptive_loop():
    """Exercise the new carry leaf in the compiled adaptive solver loop."""

    solver = RADAUSolver(
        t0=0.0,
        t1=1.0e-2,
        dt=1.0e-3,
        rtol=1.0e-6,
        atol=1.0e-8,
        max_steps=64,
        controller_mode="hairer_lean_transport_discounted_newton_bracket",
    )
    out = solver.solve(jnp.asarray([1.0]), lambda _t, y: -y)
    assert int(out["n_steps"]) > 0
    assert jnp.all(jnp.isfinite(out["final_state"]))


def test_flat_support_pullback_forwards_local_vjp_primal_reuse_flag():
    """The isolated rebuild mode must reach the NTX support hook unchanged."""

    observed = {}

    class _Owner:
        def vector_field(self, state):
            return state

        def pullback_build_lagged_response_support_payload(
            self,
            state,
            lagged_response_bar,
            support,
            **kwargs,
        ):
            observed["state"] = state
            observed["lagged_response_bar"] = lagged_response_bar
            observed["support"] = support
            observed["reuse"] = kwargs["reuse_local_vjp_primal_anchor_response"]
            observed["support_only"] = kwargs["support_only_ntx_implicit_pullback"]
            observed["profile_annotations"] = kwargs["reverse_segment_profile_annotations"]
            observed["inner_timing_component"] = kwargs[
                "reverse_rebuild_inner_timing_component"
            ]
            return {"x": jnp.asarray(3.0)}

    owner = _Owner()
    pullback = transport_solvers._flat_rhs_build_support_pullback_factory(
        lambda flat_state: flat_state,
        owner.vector_field,
        (),
        {},
    )
    result = pullback(
        jnp.asarray([2.0]),
        jnp.asarray([5.0]),
        {"x": jnp.asarray(7.0)},
        reverse_segment_profile_annotations_override=True,
        reuse_local_vjp_primal_anchor_response=True,
        support_only_ntx_implicit_pullback=True,
        reverse_rebuild_inner_timing_component="local_ntx_vjp_and_accumulation",
    )

    assert jnp.allclose(result["x"], jnp.asarray(3.0))
    assert observed["reuse"] is True
    assert observed["support_only"] is True
    assert observed["profile_annotations"] is True
    assert observed["inner_timing_component"] == "local_ntx_vjp_and_accumulation"
    assert jnp.allclose(observed["state"], jnp.asarray([2.0]))
    assert jnp.allclose(observed["lagged_response_bar"], jnp.asarray([5.0]))
    assert jnp.allclose(observed["support"]["x"], jnp.asarray(7.0))


def test_flat_support_pullback_omits_inner_timing_selector_by_default():
    """The diagnostic selector must not alter the normal model-hook contract."""

    observed = {}

    class _Owner:
        def vector_field(self, state):
            return state

        def pullback_build_lagged_response_support_payload(
            self,
            state,
            lagged_response_bar,
            support,
            **kwargs,
        ):
            del state, lagged_response_bar, support
            observed.update(kwargs)
            return {"x": jnp.asarray(0.0)}

    owner = _Owner()
    pullback = transport_solvers._flat_rhs_build_support_pullback_factory(
        lambda flat_state: flat_state,
        owner.vector_field,
        (),
        {},
    )
    pullback(jnp.asarray([1.0]), jnp.asarray([2.0]), {"x": jnp.asarray(3.0)})

    assert "reverse_rebuild_inner_timing_component" not in observed
    assert "support_only_ntx_implicit_pullback" not in observed


def test_segment_primal_record_bwd_avoids_second_minimal_attempt_reconstruction(monkeypatch):
    """The record route must consume the replayed primal, not reconstruct it again.

    This is deliberately a tiny routing test: the full numerical direct-adjoint
    contract is covered elsewhere, while this protects the exact structural
    property relevant to reverse cost.  No Radau/NTX solve is performed.
    """

    observed = {"minimal_attempt_calls": 0, "record_adapter_calls": 0}
    sentinel_primal = object()

    def _minimal_attempt(*_args):
        observed["minimal_attempt_calls"] += 1
        return sentinel_primal

    def _record_adapter(*_args):
        observed["record_adapter_calls"] += 1
        return sentinel_primal

    def _from_primal(*args):
        assert args[5] is sentinel_primal
        return "reduced-bars", ("support-bars",)

    monkeypatch.setattr(
        transport_solvers,
        "_execute_radau_accepted_step_attempt_reverse_minimal",
        _minimal_attempt,
    )
    monkeypatch.setattr(
        transport_solvers,
        "_radau_reverse_minimal_attempt_from_segment_primal_record",
        _record_adapter,
    )
    monkeypatch.setattr(
        transport_solvers,
        "_execute_radau_accepted_step_next_reduced_cotangent_batched_bwd_with_support_from_primal_result",
        _from_primal,
    )

    common_args = (object(), object(), object(), "rebuild", object(), object(), object())
    reconstruct_result = (
        transport_solvers._execute_radau_accepted_step_next_reduced_cotangent_batched_bwd_with_support(
            *common_args
        )
    )
    record_result = (
        transport_solvers._execute_radau_accepted_step_next_reduced_cotangent_batched_bwd_with_support_from_segment_primal_record(
            object(), object(), object(), "rebuild", object(), object(), object(), object()
        )
    )

    assert reconstruct_result == record_result == ("reduced-bars", ("support-bars",))
    assert observed == {"minimal_attempt_calls": 1, "record_adapter_calls": 1}


def test_radau_controller_keeps_the_moderate_cap_on_an_easy_accepted_step():
    _, info = _apply_radau_lean_timestep_controller(**_radau_controller_arguments(newton_iter_count=3))
    assert float(info.growth) == pytest.approx(1.5)


def test_radau_joint_rebuild_support_pullback_accumulates_batched_bars(monkeypatch):
    """Exercise the Radau rebuild branch that dispatches the joint NTX hook.

    The stage residual kernels are deliberately zeroed here: their numerical
    transpose is covered separately.  This test protects the integration
    contract that previously failed only in the full transport benchmark:
    batched rebuild bars must reach the joint state+support pullback, then be
    accumulated into the reduced carry and the support cotangent leaves.
    """
    dtype = jnp.float64
    y = jnp.zeros((1,), dtype=dtype)
    zero_int = jnp.asarray(0, dtype=jnp.int32)
    carry = transport_solvers._RadauAcceptedStepCarry(
        t=jnp.asarray(0.0, dtype=dtype),
        y=y,
        dt=jnp.asarray(1.0, dtype=dtype),
        prev_error=jnp.asarray(0.0, dtype=dtype),
        prev_stages=jnp.zeros((3, 1), dtype=dtype),
        prev_dt=jnp.asarray(1.0, dtype=dtype),
        recent_reject_count=zero_int,
        regrowth_cooldown=zero_int,
        easy_growth_streak=zero_int,
        lagged_response_cache=jnp.asarray(0.0, dtype=dtype),
        lagged_response_valid=jnp.asarray(True),
        lagged_reference_y=y,
        jacobian=jnp.zeros((1, 1), dtype=dtype),
        cache_valid=jnp.asarray(True),
        cache_dt=jnp.asarray(1.0, dtype=dtype),
        cache_age=zero_int,
        real_lu=jnp.eye(1, dtype=dtype),
        real_piv=jnp.zeros((1,), dtype=jnp.int32),
        complex_lu=jnp.eye(2, dtype=dtype),
        complex_piv=jnp.zeros((2,), dtype=jnp.int32),
        prev_theta_final=jnp.asarray(0.0, dtype=dtype),
        prev_newton_iter_count=zero_int,
    )
    primal_result = transport_solvers._RadauAcceptedStepReverseMinimalAttemptResult(
        carry_after_attempt=carry,
        trial_dt=jnp.asarray(1.0, dtype=dtype),
        trial_y=y,
        stage_history=jnp.zeros((3, 1), dtype=dtype),
        jacobian_out=jnp.zeros((1, 1), dtype=dtype),
        cache_valid_out=jnp.asarray(True),
        cache_dt_out=jnp.asarray(1.0, dtype=dtype),
        cache_age_out=zero_int,
        real_lu_out=jnp.eye(1, dtype=dtype),
        real_piv_out=jnp.zeros((1,), dtype=jnp.int32),
        complex_lu_out=jnp.eye(2, dtype=dtype),
        complex_piv_out=jnp.zeros((2,), dtype=jnp.int32),
        theta_final=jnp.asarray(0.0, dtype=dtype),
        newton_iter_count=zero_int,
    )

    monkeypatch.setattr(
        transport_solvers,
        "_radau_prepare_lagged_response",
        lambda *args, **kwargs: (jnp.asarray(1.0, dtype=dtype), jnp.asarray(False), jnp.asarray(False)),
    )
    monkeypatch.setattr(
        transport_solvers,
        "_radau_solve_exact_stage_residual_transpose_batched",
        lambda *args, **kwargs: jnp.zeros_like(kwargs["rhs"]),
    )
    monkeypatch.setattr(
        transport_solvers,
        "_radau_exact_stage_residual_input_pullback",
        lambda *args, **kwargs: (
            jnp.zeros_like(y),
            jnp.asarray(0.0, dtype=dtype),
            jnp.asarray(0.0, dtype=dtype),
        ),
    )
    monkeypatch.setattr(
        transport_solvers,
        "_radau_exact_stage_residual_support_pullback",
        lambda *args, **kwargs: {"x": jnp.asarray(0.0, dtype=dtype)},
    )

    def joint_rebuild_pullback(state, lagged_response_bars, support):
        assert state.shape == (1,)
        assert support["x"].shape == ()
        return 3.0 * lagged_response_bars[:, None], {"x": 11.0 * lagged_response_bars}

    physics_context = types.SimpleNamespace(
        reverse_direct_stage_adjoint=True,
        reverse_rhs_pullback_mode="separate",
        reverse_stage_cotangent_mode="full",
        reverse_stage_adjoint_memory_mode="default",
        reverse_rebuild_support_pullback_mode="ntx_joint_implicit_interpolated_faces",
        reverse_segment_profile_annotations=False,
        pullback_build_lagged_response=object(),
        unpack_flat=lambda value: value,
        project_flat=None,
        build_lagged_response=object(),
        flat_rhs_build_support_pullback=None,
        flat_rhs_build_state_and_support_pullback_batched_interpolated_faces=joint_rebuild_pullback,
    )
    kernel_context = types.SimpleNamespace(dtype=dtype, b=jnp.ones((3,), dtype=dtype))
    next_bars = transport_solvers._RadauAcceptedStepReducedCotangent(
        y=jnp.asarray([[10.0], [20.0]], dtype=dtype),
        lagged_response_cache=jnp.asarray([2.0, -1.0], dtype=dtype),
        lagged_reference_y=jnp.zeros((2, 1), dtype=dtype),
    )

    reduced_bars, support_leaves = (
        transport_solvers._execute_radau_accepted_step_next_reduced_cotangent_batched_bwd_with_support_from_primal_result(
            kernel_context,
            physics_context,
            types.SimpleNamespace(),
            "rebuild",
            carry,
            primal_result,
            next_bars,
            {"x": jnp.asarray(0.0, dtype=dtype)},
        )
    )

    assert jnp.allclose(reduced_bars.y, jnp.asarray([[16.0], [17.0]], dtype=dtype))
    assert jnp.allclose(reduced_bars.lagged_response_cache, jnp.zeros((2,), dtype=dtype))
    assert jnp.allclose(reduced_bars.lagged_reference_y, jnp.zeros((2, 1), dtype=dtype))
    assert len(support_leaves) == 1
    assert jnp.allclose(support_leaves[0], jnp.asarray([22.0, -11.0], dtype=dtype))

    # The new selector must use its dedicated context hook, while preserving
    # the same combined rebuild-state/support contract.
    physics_context.reverse_rebuild_support_pullback_mode = (
        "ntx_joint_implicit_interpolated_faces_reuse_local_vjp_primal"
    )
    physics_context.flat_rhs_build_state_and_support_pullback_batched_interpolated_faces_reuse_local_vjp_primal = (
        joint_rebuild_pullback
    )
    reused_reduced_bars, reused_support_leaves = (
        transport_solvers._execute_radau_accepted_step_next_reduced_cotangent_batched_bwd_with_support_from_primal_result(
            kernel_context,
            physics_context,
            types.SimpleNamespace(),
            "rebuild",
            carry,
            primal_result,
            next_bars,
            {"x": jnp.asarray(0.0, dtype=dtype)},
        )
    )
    assert jnp.allclose(reused_reduced_bars.y, reduced_bars.y)
    assert jnp.allclose(reused_support_leaves[0], support_leaves[0])
