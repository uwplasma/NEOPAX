import types

import jax.numpy as jnp
import pytest

import NEOPAX._transport_solvers as transport_solvers

from NEOPAX._transport_solvers import (
    DiffraxSolver,
    NewtonThetaMethodSolver,
    RADAUSolver,
    ThetaMethodSolver,
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


def test_build_time_solver_theta_newton_backend():
    pytest.importorskip("diffrax")
    solver = build_time_solver(_base_solver_parameters(transport_solver_backend="theta_newton"))
    assert isinstance(solver, NewtonThetaMethodSolver)
    assert float(solver.t0) == 0.0
    assert float(solver.t1) == 1.0
    assert solver.rhs_mode == "black_box"


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
            observed["profile_annotations"] = kwargs["reverse_segment_profile_annotations"]
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
    )

    assert jnp.allclose(result["x"], jnp.asarray(3.0))
    assert observed["reuse"] is True
    assert observed["profile_annotations"] is True
    assert jnp.allclose(observed["state"], jnp.asarray([2.0]))
    assert jnp.allclose(observed["lagged_response_bar"], jnp.asarray([5.0]))
    assert jnp.allclose(observed["support"]["x"], jnp.asarray(7.0))


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
