import jax.numpy as jnp
import pytest

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


def test_radau_controller_keeps_the_moderate_cap_on_an_easy_accepted_step():
    _, info = _apply_radau_lean_timestep_controller(**_radau_controller_arguments(newton_iter_count=3))
    assert float(info.growth) == pytest.approx(1.5)
