"""Scalar synthetic Radau replay/guard regressions; no transport or VMEC/NTX."""

import dataclasses
import ast
import inspect
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from NEOPAX import _transport_solvers as solvers
from NEOPAX._optimization_full_transport_stage import _validate_database_primal_replay
from NEOPAX._reverse_step_diagnostics import check_primal_replay_endpoint
from NEOPAX import _reverse_ad_transport as reverse


def _core_replay_nodes():
    source = inspect.getsource(reverse.realtime_geometry_reverse_all_objectives_support_payload_bar_for_parameter_vector)
    return ast.parse(source).body[0].body


def _benchmark_segment_replay():
    function = next(node for node in _core_replay_nodes()
                    if isinstance(node, ast.FunctionDef) and node.name == "_segment_replay_minimal_with_primal_records")
    namespace = {"jax": jax, "np": np,
                 "dependencies": SimpleNamespace(segment_replay_minimal_with_primal_records=None),
                 "_radau_segment_replay_minimal_with_primal_records_call":
                     solvers._radau_segment_replay_minimal_with_primal_records_call}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<benchmark-replay>", "exec"), namespace)
    return namespace[function.name]


def test_benchmark_reuse_selector_filters_rejected_attempts_and_preserves_production_default():
    body = _core_replay_nodes()
    selection = next(node for node in body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "replay_uses_forward_setup"
                             for target in node.targets))
    start = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "forward_rebuild_mask"
                         for target in node.targets))
    code = compile(ast.Module(body=[selection, *body[start:start + 3]], type_ignores=[]), "<replay-mask>", "exec")
    trace = SimpleNamespace(active_mask=np.array([True, True, True, True, False]),
                            accepted_mask=np.array([True, False, True, True, False]),
                            jacobian_reused=np.array([False, False, True, False, False]))
    for mode, adapter in (("legacy", None), ("recorded_forward", None), (None, None), (None, object())):
        namespace = {"np": np, "jax": jax, "replay_jacobian_reuse_mode": mode,
                     "dependencies": SimpleNamespace(segment_replay_minimal_with_primal_records=adapter),
                     "schedule_artifact": trace, "segmented_replay_arrays": (np.zeros((2, 2)),)}
        exec(code, namespace)
        if mode == "recorded_forward" or adapter is not None:
            np.testing.assert_array_equal(namespace["forward_rebuild_mask"], [[True, False], [True, False]])
        else:
            assert namespace["forward_rebuild_mask"] is None
    namespace["replay_jacobian_reuse_mode"] = "recorded_forward"
    namespace["schedule_artifact"] = None
    with pytest.raises(ValueError, match="requires the recorded forward Newton reuse decisions"):
        exec(code, namespace)


def test_benchmark_checked_replay_uses_existing_scalar_kernel_without_optimization_stages():
    replay = _benchmark_segment_replay()
    context, carry = _case()
    schedule = _schedule([True, True, False])
    legacy = replay(context, carry, schedule)
    checked = replay(context, carry, schedule, force_jacobian_rebuild=jnp.array([True, False, False]))
    assert legacy[2].replay_valid is None
    np.testing.assert_array_equal(checked[2].replay_valid, [True, True, True])
    for a, b in zip(jax.tree_util.tree_leaves(legacy[:2]), jax.tree_util.tree_leaves(checked[:2]), strict=True):
        np.testing.assert_array_equal(a, b)


def test_benchmark_recorded_replay_rejects_failed_scalar_reconstruction_before_ad():
    replay = _benchmark_segment_replay()
    context, carry = _case()
    poisoned = dataclasses.replace(carry, prev_stages=jnp.full_like(carry.prev_stages, 1e200),
                                   prev_dt=jnp.asarray(.01))
    with pytest.raises(FloatingPointError, match="Invalid primal replay at slot=0"):
        replay(context, poisoned, _schedule([True, True]), force_jacobian_rebuild=jnp.array([True, False]))


def _case():
    solver = solvers.RADAUSolver(
        t0=0., t1=.2, dt=.01, rtol=1e-6, atol=1e-10,
        num_stages=3, maxiter=8, max_steps=8, rhs_mode="black_box",
    )
    prepared = solvers._build_prepared_radau_accepted_rollout(
        solver=solver, state=jnp.array([1.]), vector_field=lambda t, y, *args: -(y ** 2),
        species=None,
    )
    context = solvers._build_prepared_radau_execution_context(solver=solver, prepared_rollout=prepared)
    return context, prepared.initial_carry


def _schedule(active):
    active = jnp.asarray(active)
    count = active.shape[0]
    return (active, jnp.full(count, .01), jnp.full(count, .01),
            jnp.zeros(count, dtype=jnp.int32), jnp.zeros(count, dtype=jnp.int32),
            jnp.zeros(count, dtype=jnp.int32), jnp.zeros(count, dtype=bool))


def test_failed_finite_history_is_reported_without_changing_the_scalar_solve():
    context, carry = _case()
    poisoned = dataclasses.replace(carry, prev_stages=jnp.full_like(carry.prev_stages, 1e200),
                                   prev_dt=jnp.asarray(.01))
    initial = solvers._execute_radau_accepted_step_attempt_reverse_minimal(
        context.kernel_context, context.physics_context, poisoned, context.attempt_context,
        return_convergence=True,
    )
    assert not bool(initial.converged)
    checked = jax.jit(lambda value: solvers._radau_checked_primal_replay_record(context, value))(poisoned)
    assert not bool(checked.replay_valid)
    assert not bool(checked.replay_converged)
    assert float(checked.trial_dt) == float(carry.dt)
    np.testing.assert_allclose(checked.trial_y, initial.trial_y, equal_nan=True)
    np.testing.assert_allclose(checked.stage_history, initial.stage_history, equal_nan=True)


def test_valid_scalar_segment_matches_physical_solution_and_masks_padding():
    context, carry = _case()
    schedule = _schedule([True, True, False])
    result = solvers._radau_segment_replay_minimal_with_primal_records_call(
        context, carry, schedule, validate_primal=True,
    )
    _validate_database_primal_replay(result, schedule)
    final, _, records = result
    np.testing.assert_array_equal(records.replay_valid, [True, True, True])
    np.testing.assert_array_equal(records.replay_converged, [True, True, True])
    # y'=-y^2, y(0)=1 has exact solution 1/(1+t).
    np.testing.assert_allclose(final.y, [1 / 1.02], rtol=1e-6, atol=1e-10)
    assert float(final.t) == pytest.approx(.02)


def test_unconverged_finite_result_is_rejected_before_ad_and_later_slots_stop(monkeypatch, tmp_path):
    context, carry = _case()
    good = solvers._execute_radau_accepted_step_attempt_reverse_minimal(
        context.kernel_context, context.physics_context, carry, context.attempt_context,
        return_convergence=True,
    )
    bad = dataclasses.replace(good, converged=jnp.asarray(False), final_residual_norm=jnp.asarray(10.))
    monkeypatch.setattr(solvers, "_execute_radau_accepted_step_attempt_reverse_minimal", lambda *args, **kwargs: bad)
    schedule = _schedule([True, True])
    result = solvers._radau_segment_replay_minimal_with_primal_records_call(
        context, carry, schedule, validate_primal=True,
    )
    final, starts, records = result
    np.testing.assert_array_equal(records.replay_valid, [False, True])
    np.testing.assert_array_equal(records.replay_converged, [False, True])
    np.testing.assert_array_equal(final.y, carry.y)
    np.testing.assert_array_equal(starts.y[1], carry.y)
    assert float(final.t) == float(carry.t)
    monkeypatch.setenv("NEOPAX_REVERSE_STEP_SNAPSHOT_DIR", str(tmp_path))
    with pytest.raises(FloatingPointError, match="invalid primal replay at slot=0"):
        _validate_database_primal_replay(result, schedule)
    snapshots = list(tmp_path.glob("*.npz"))
    assert len(snapshots) == 1
    with np.load(snapshots[0]) as data:
        assert str(data["diagnostic_stage"]) == "primal_replay_failed"
        assert not bool(data["replay_valid"])


def test_checking_healthy_replay_preserves_existing_primal_records():
    context, carry = _case()
    schedule = _schedule([True, True, False])
    legacy = solvers._radau_segment_replay_minimal_with_primal_records_call(
        context, carry, schedule,
    )
    checked = solvers._radau_segment_replay_minimal_with_primal_records_call(
        context, carry, schedule, validate_primal=True,
    )
    for a, b in zip(jax.tree_util.tree_leaves(legacy[:2]), jax.tree_util.tree_leaves(checked[:2]), strict=True):
        np.testing.assert_array_equal(a, b)
    for field in dataclasses.fields(legacy[2]):
        if not field.name.startswith("replay_"):
            np.testing.assert_array_equal(getattr(legacy[2], field.name), getattr(checked[2], field.name))


def test_endpoint_guard_rejects_finite_different_trajectory_and_nonfinite_output():
    peak, scaled = check_primal_replay_endpoint(jnp.array([1. + 1e-7]), jnp.array([1.]), atol=1e-6, rtol=1e-6)
    assert peak > 0 and scaled < 1
    for value in (1.01, np.nan):
        with pytest.raises(FloatingPointError, match="endpoint differs from adaptive checkpoint"):
            check_primal_replay_endpoint(jnp.array([value]), jnp.array([1.]), atol=1e-6, rtol=1e-6)


def test_recorded_rebuild_restores_forward_newton_setup_after_rejected_attempt():
    solver = solvers.RADAUSolver(
        t0=0., t1=1., dt=.001, rtol=.01, atol=1e-8,
        num_stages=3, maxiter=12, max_steps=8, rhs_mode="black_box",
        jacobian_reuse_mode="legacy", predictor_mode="collocation_transport_weighted",
    )
    prepared = solvers._build_prepared_radau_accepted_rollout(
        solver=solver, state=jnp.array([1.]), vector_field=lambda t, y, *args: -100. * y ** 3,
        species=None,
    )
    context = solvers._build_prepared_radau_execution_context(solver=solver, prepared_rollout=prepared)
    step = jax.jit(lambda value: solvers._radau_step_fn(context, value, None))
    initial = solvers._radau_step_state_from_carry(
        prepared.initial_carry, status=jnp.array([0, 0, 0], dtype=jnp.int32),
    )
    first_state, first = step(initial)
    rejected_state, rejected = step(dataclasses.replace(first_state, dt=jnp.asarray(.05)))
    final_state, last = step(dataclasses.replace(rejected_state, dt=jnp.asarray(.001)))
    assert bool(first.accepted) and not bool(rejected.accepted) and bool(last.accepted)
    np.testing.assert_array_equal(first_state.y, rejected_state.y)
    schedule = list(_schedule([True, True]))
    schedule[1] = schedule[2] = jnp.full(2, .001)
    schedule.append(jnp.array([~first.jacobian_reused, ~last.jacobian_reused]))
    final, _, records = solvers._radau_segment_replay_minimal_with_primal_records_call(
        context, prepared.initial_carry, tuple(schedule), validate_primal=True,
    )
    assert bool(jnp.all(records.replay_valid))
    np.testing.assert_allclose(records.jacobian_out[-1], final_state.jacobian, rtol=1e-13)
    np.testing.assert_allclose(final.y, final_state.y, rtol=1e-13)
    assert int(records.newton_iter_count[-1]) == int(last.newton_iter_count)


def test_adaptive_schedule_can_retain_scalar_forward_newton_decisions():
    context, carry = _case()
    result = solvers._radau_adaptive_schedule_rollout(
        context, carry, max_total_steps=8, stop_after_accepted_steps=2,
        capture_segment_length=2, capture_newton_setup=True,
    )
    assert int(result.accepted_count) == 2
    assert result.trace.jacobian_reused.shape == result.trace.active_mask.shape == (8,)
    assert not bool(result.trace.jacobian_reused[0])
    assert result.trace.jacobian_reused.dtype == jnp.dtype(bool)
