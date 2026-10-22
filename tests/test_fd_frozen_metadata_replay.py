"""FD replay regressions on one-variable ODEs; no VMEC, NTX or transport."""

import ast
import dataclasses
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from NEOPAX import _transport_solvers as solvers


BENCHMARK = Path(__file__).resolve().parents[1] / (
    "examples/benchmarks/benchmark_transport_realtime_geometry_forward_fd.py"
)


def _namespace():
    names = {
        "_tree_all_finite", "_fd_reverse_frozen_metadata_accepted_rollout",
        "_require_complete_frozen_metadata_replay", "_schedule_rollout",
        "_save_fd_replay_failure_snapshot",
    }
    functions = [node for node in ast.parse(BENCHMARK.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {
        "jax": jax, "jnp": jnp, "np": np, "dataclasses": dataclasses,
        "Path": Path, "json": json, "tempfile": tempfile,
        "_RadauAcceptedRolloutResult": solvers._RadauAcceptedRolloutResult,
        "_RadauAcceptedStepAttemptContext": solvers._RadauAcceptedStepAttemptContext,
        "_radau_apply_accepted_step_map": solvers._radau_apply_accepted_step_map,
    }
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *functions,
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(BENCHMARK), "exec"), namespace)
    return namespace


def _prepared(rate=2.0):
    solver = solvers.RADAUSolver(
        t0=0.0, t1=0.03, dt=0.01, rtol=1e-7, atol=1e-10,
        num_stages=3, maxiter=8, max_steps=8, rhs_mode="black_box",
        jacobian_reuse_mode="legacy",
    )
    return solvers._build_prepared_radau_accepted_rollout(
        solver=solver, state=jnp.array([1.0]),
        vector_field=lambda t, y, *args: -rate * y, species=None,
    )


def _trace():
    # Slot 1 is rejected, slot 4 inactive: neither is a replayed step.
    return SimpleNamespace(
        active_mask=jnp.array([True, True, True, True, False]),
        accepted_mask=jnp.array([True, False, True, True, False]),
        attempted_dts=jnp.array([0.01, 0.7, 0.01, 0.01, 0.0]),
        next_dts=jnp.full(5, 0.01),
        next_recent_reject_count=jnp.zeros(5, dtype=jnp.int32),
        next_regrowth_cooldown=jnp.zeros(5, dtype=jnp.int32),
        next_easy_growth_streak=jnp.zeros(5, dtype=jnp.int32),
        next_lagged_response_valid=jnp.zeros(5, dtype=bool),
        jacobian_reused=jnp.array([False, False, True, False, False]),
    )


def _run(namespace, baseline, endpoint=None, trace=None, **kwargs):
    endpoint = baseline if endpoint is None else endpoint
    return namespace["_fd_reverse_frozen_metadata_accepted_rollout"](
        baseline_prepared_rollout=baseline, endpoint_prepared_rollout=endpoint,
        trace=_trace() if trace is None else trace,
        endpoint_carry0=endpoint.initial_carry, endpoint_label="n0:minus",
        **kwargs,
    )


def test_full_scalar_fd_matches_exact_derivative_and_ignores_rejected_padding():
    namespace = _namespace()
    baseline = _prepared()
    step = 1e-4
    minus = _run(namespace, baseline, _prepared(2.0 - step))
    plus = _run(namespace, baseline, _prepared(2.0 + step))
    namespace["_require_complete_frozen_metadata_replay"](
        rollout=minus, prepared_rollout=baseline, trace=_trace(),
    )
    np.testing.assert_array_equal(minus.converged_mask, [True, False, True, True, False])
    assert float(minus.final_carry.t) == pytest.approx(0.03)
    np.testing.assert_allclose(minus.final_y, np.exp(-(2.0 - step) * 0.03), rtol=1e-10)
    fd = (np.asarray(plus.final_y) - np.asarray(minus.final_y)) / (2 * step)
    np.testing.assert_allclose(fd, -0.03 * np.exp(-2.0 * 0.03), rtol=1e-7)


def _map_result(carry, *, y, converged):
    next_carry = dataclasses.replace(
        carry, t=carry.t + carry.dt, y=y, cache_valid=jnp.asarray(True),
    )
    return solvers._RadauAcceptedStepMapResult(
        next_carry=next_carry, accepted_y=y, accepted_t=next_carry.t,
        accepted_dt=carry.dt, err_norm=jnp.asarray(0.1),
        converged=converged, stage_history=carry.prev_stages,
        theta_final=jnp.asarray(0.0), newton_iter_count=jnp.asarray(2, dtype=jnp.int32),
        final_residual_norm=jnp.asarray(0.0), final_delta_norm=jnp.asarray(0.0),
        density_err_norm=jnp.asarray(0.0), pressure_err_norm=jnp.asarray(0.0),
        er_err_norm=jnp.asarray(0.0),
    )


def test_recorded_rebuilds_clear_both_replay_caches_on_accepted_slots_only():
    namespace = _namespace()
    prepared = _prepared()
    prepared = dataclasses.replace(
        prepared, initial_carry=dataclasses.replace(
            prepared.initial_carry, cache_valid=jnp.asarray(True),
        ),
    )

    def map_step(kernel, physics, carry, context):
        # Encode each cache decision in the output to observe the real scan.
        y = carry.y + jnp.where(carry.cache_valid, 1.0, 10.0)
        return _map_result(carry, y=y, converged=jnp.asarray(True))

    namespace["_radau_apply_accepted_step_map"] = map_step
    result = _run(namespace, prepared)
    # Rebuild, reuse, rebuild: +10 +1 +10. The rejected +0.7 slot is absent.
    np.testing.assert_array_equal(result.final_y, [22.0])
    assert float(result.final_carry.t) == pytest.approx(0.03)


@pytest.mark.parametrize("role", ["baseline", "perturbed_endpoint"])
@pytest.mark.parametrize("failure", ["unconverged", "nonfinite"])
def test_first_bad_accepted_step_stops_before_bootstrap(role, failure):
    namespace = _namespace()
    baseline, endpoint = _prepared(), _prepared(2.001)

    def map_step(kernel, physics, carry, context):
        target = baseline if role == "baseline" else endpoint
        bad = jnp.asarray(physics is target.physics_context) & (carry.t >= 0.009)
        y = jnp.where(bad, jnp.nan, carry.y + 1.0) if failure == "nonfinite" else carry.y + 1.0
        converged = ~bad if failure == "unconverged" else jnp.asarray(True)
        return _map_result(carry, y=y, converged=converged)

    namespace["_radau_apply_accepted_step_map"] = map_step
    with pytest.raises(
        FloatingPointError,
        match=f"endpoint=n0:minus role={role} accepted_step=2 attempt_slot=2",
    ) as error:
        _run(namespace, baseline, endpoint)
    assert "Stopped before objective evaluation" in str(error.value)
    assert f"newton_converged={failure == 'nonfinite'}" in str(error.value)


def test_missing_forward_reuse_decisions_are_rejected():
    namespace = _namespace()
    trace = _trace()
    trace.jacobian_reused = None
    with pytest.raises(ValueError, match="requires recorded forward Jacobian reuse"):
        _run(namespace, _prepared(), trace=trace)


def test_cold_restart_keeps_physical_input_and_timestep_and_reports_recovered_slot():
    namespace = _namespace()
    baseline, endpoint = _prepared(), _prepared(2.001)

    def map_step(kernel, physics, carry, context):
        bad = (
            jnp.asarray(physics is endpoint.physics_context)
            & (carry.t > 0.009) & (carry.t < 0.011)
            & carry.cache_valid
        )
        # A failed trial must never become the input to the cold restart.
        y = jnp.where(bad, 1e100, carry.y + 1.0)
        result = _map_result(carry, y=y, converged=~bad)
        result = dataclasses.replace(result, next_carry=dataclasses.replace(
            result.next_carry, prev_dt=carry.dt, prev_stages=jnp.ones_like(carry.prev_stages),
        ))
        return result

    namespace["_radau_apply_accepted_step_map"] = map_step
    diagnostics = {}
    result = _run(namespace, baseline, endpoint, diagnostics_out=diagnostics)
    np.testing.assert_array_equal(result.final_y, [4.0])
    np.testing.assert_allclose(result.accepted_dts, [0.01, 0.0, 0.01, 0.01, 0.0])
    assert float(result.final_carry.t) == pytest.approx(0.03)
    assert diagnostics == {
        "accepted_steps": 3, "baseline_newton_retries": 0,
        "endpoint_newton_retries": 1, "endpoint_retry_accepted_steps": [2],
        "fixed_timesteps": True,
    }


def test_frozen_jacobian_failure_recovers_on_a_scalar_piecewise_rhs():
    namespace = _namespace()
    solver = solvers.RADAUSolver(
        t0=0.0, t1=0.01, dt=0.01, rtol=1e-7, atol=1e-10,
        num_stages=3, maxiter=4, max_steps=4, rhs_mode="black_box",
        jacobian_reuse_mode="legacy",
    )
    prepared = solvers._build_prepared_radau_accepted_rollout(
        solver=solver, state=jnp.array([-1e-4]),
        vector_field=lambda t, y, *args: -100.0 * jnp.maximum(y, 0.0), species=None,
    )
    context = solvers._RadauAcceptedStepAttemptContext(
        t_final=jnp.asarray(0.01), use_transport_lagged_response=jnp.asarray(False),
    )
    warm = solvers._radau_apply_accepted_step_map(
        prepared.kernel_context, prepared.physics_context, prepared.initial_carry, context,
    )
    assert bool(warm.converged)
    baseline = dataclasses.replace(prepared, initial_carry=dataclasses.replace(
        warm.next_carry, t=jnp.asarray(0.0),
    ))
    endpoint = dataclasses.replace(prepared, initial_carry=dataclasses.replace(
        baseline.initial_carry, y=jnp.array([1e-4]),
    ))
    # The frozen baseline has a zero Jacobian on the negative branch.
    frozen_trial = solvers._radau_apply_accepted_step_map(
        endpoint.kernel_context, endpoint.physics_context, endpoint.initial_carry, context,
    )
    assert not bool(frozen_trial.converged)
    trace = _trace()
    for key in vars(trace):
        setattr(trace, key, getattr(trace, key)[:1])
    trace.jacobian_reused = jnp.array([True])
    diagnostics = {}
    result = _run(namespace, baseline, endpoint, trace=trace, diagnostics_out=diagnostics)
    assert diagnostics["endpoint_newton_retries"] == 1
    assert diagnostics["baseline_newton_retries"] == 0
    assert bool(result.converged_mask[0])
    assert float(result.final_carry.t) == pytest.approx(0.01)
    # The returned metadata is baseline-frozen; verify the physical endpoint
    # independently with the three-stage Radau stability function at z=-1.
    a = np.asarray(endpoint.kernel_context.a)
    b = np.asarray(endpoint.kernel_context.b)
    linear_stages = np.linalg.solve(np.eye(3) + a, np.ones(3))
    assert np.all(linear_stages > 0.0)
    stability = 1.0 - b @ linear_stages
    np.testing.assert_allclose(result.final_y, [1e-4 * stability], rtol=1e-8, atol=1e-13)


@pytest.mark.parametrize("failure", ["nonfinite", "unconverged", "incomplete"])
def test_final_time_alone_cannot_validate_fd_replay(failure):
    namespace = _namespace()
    prepared = _prepared()
    result = solvers._RadauAcceptedRolloutResult(
        final_carry=dataclasses.replace(prepared.initial_carry, t=jnp.asarray(0.03)),
        final_y=jnp.array([1.0]), trial_ys=jnp.ones((5, 1)),
        err_norms=jnp.zeros(5), converged_mask=_trace().accepted_mask,
        accepted_dts=jnp.where(_trace().accepted_mask, 0.01, 0.0),
    )
    if failure == "nonfinite":
        result = dataclasses.replace(result, final_y=jnp.array([jnp.nan]))
    elif failure == "unconverged":
        result = dataclasses.replace(result, converged_mask=result.converged_mask.at[2].set(False))
    else:
        result = dataclasses.replace(result, final_carry=dataclasses.replace(result.final_carry, t=jnp.asarray(0.02)))
    with pytest.raises((FloatingPointError, RuntimeError)):
        namespace["_require_complete_frozen_metadata_replay"](
            rollout=result, prepared_rollout=prepared, trace=_trace(),
        )


def test_schedule_records_reuse_without_prefix_or_checkpoint_bank():
    namespace = _namespace()
    prepared = _prepared()
    solver = SimpleNamespace(max_steps=20000, t1=0.03, stop_after_accepted_steps=None)
    rollout = SimpleNamespace(
        final_carry=dataclasses.replace(prepared.initial_carry, t=jnp.asarray(0.03)),
        failed=False, completed=True, fail_code=0, accepted_count=3, attempt_count=4,
    )
    namespace["_prepare_rollout"] = lambda *_args: (
        {"solver": solver}, prepared, object(),
    )
    namespace["_radau_adaptive_schedule_rollout"] = Mock(return_value=rollout)
    namespace["_schedule_rollout"]({}, object(), object(), accepted_step_limit=None,
                                   capture_newton_setup=True)
    kwargs = namespace["_radau_adaptive_schedule_rollout"].call_args.kwargs
    assert kwargs == {
        "max_total_steps": 20000, "stop_after_accepted_steps": None,
        "capture_newton_setup": True,
    }


def _probe_module():
    path = BENCHMARK.with_name("diagnose_transport_fd_replay_step.py")
    spec = importlib.util.spec_from_file_location("fd_saved_step_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _failed_snapshot(tmp_path):
    namespace = _namespace()
    baseline, endpoint = _prepared(), _prepared(2.001)

    def map_step(kernel, physics, carry, context):
        bad = jnp.asarray(physics is endpoint.physics_context) & (carry.t > 0.009)
        return _map_result(carry, y=jnp.where(bad, 1e100, carry.y + 1.0), converged=~bad)

    namespace["_radau_apply_accepted_step_map"] = map_step
    diagnostics = {}
    with pytest.raises(FloatingPointError, match="accepted_step=2"):
        _run(namespace, baseline, endpoint, diagnostics_out=diagnostics)
    failure = diagnostics["failed_step"]
    # These must be incoming states before the failed trial, not its output.
    assert float(failure["baseline_carry"].t) == pytest.approx(0.01)
    np.testing.assert_array_equal(failure["endpoint_carry"].y, [2.0])
    path = namespace["_save_fd_replay_failure_snapshot"](
        failure=failure,
        config={"case_path": Path("case.toml"), "transport_solver": {"t_final": 20.0}},
        profile_values=jnp.array([4.21, 17.8, 10.0, 2.0, 1.0, 1.0]),
        endpoint_label="n0:minus", directory=tmp_path,
    )
    return path, baseline


def test_failed_step_snapshot_roundtrip_preserves_incoming_carries_and_no_history(tmp_path):
    path, prepared = _failed_snapshot(tmp_path)
    probe = _probe_module()
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata_json"].item())
        assert metadata["config"]["case_path"] == "case.toml"
        assert metadata["failure"]["accepted_step"] == 2
        assert metadata["failure"]["attempt_slot"] == 2
        assert metadata["failure"]["cold_restart_failed"] is True
        assert len(metadata["carry_contracts"]) == 2
        restored = probe.restore_carry(archive, "endpoint_carry", prepared.initial_carry, metadata)
        np.testing.assert_array_equal(restored.y, [2.0])
        assert float(restored.t) == pytest.approx(0.01)
        assert float(archive["dt"]) == pytest.approx(0.01)
        expected_keys = {
            "metadata_json", "baseline_initial_y", "endpoint_initial_y",
            "dt", "force_jacobian_rebuild", "next_dt", "next_recent_reject_count",
            "next_regrowth_cooldown", "next_easy_growth_streak", "next_lagged_response_valid",
        }
        for name, paths in metadata["carry_contracts"].items():
            expected_keys.update(f"{name}_{index:04d}" for index in range(len(paths)))
        assert set(archive.files) == expected_keys


@pytest.mark.parametrize("mismatch", ["tree", "shape", "dtype"])
def test_saved_step_restore_rejects_changed_model_layout(tmp_path, mismatch):
    path, prepared = _failed_snapshot(tmp_path)
    probe = _probe_module()
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata_json"].item())
        template = prepared.initial_carry
        if mismatch == "tree":
            metadata["carry_contracts"]["endpoint_carry"][0] = "different"
        elif mismatch == "shape":
            template = dataclasses.replace(template, y=jnp.ones(2))
        else:
            template = dataclasses.replace(template, y=jnp.ones(1, dtype=jnp.float32))
        with pytest.raises(ValueError, match="differs from the rebuilt model"):
            probe.restore_carry(archive, "endpoint_carry", template, metadata)


def test_saved_step_probe_solves_only_current_scalar_step(monkeypatch):
    probe = _probe_module()
    prepared = _prepared()
    baseline = prepared.initial_carry
    endpoint = dataclasses.replace(baseline, y=baseline.y + 1e-6)
    monkeypatch.setattr(
        solvers, "_radau_adaptive_schedule_rollout",
        lambda *_args, **_kwargs: pytest.fail("saved-step probe entered adaptive rollout"),
    )
    report = probe.diagnose_saved_step(
        {"failure": {"accepted_step": 83}, "endpoint_label": "n0:minus"},
        baseline, endpoint, prepared, prepared, jnp.asarray(0.01), jnp.asarray(False),
    )
    assert set(report["trials"]) == {
        "baseline_recorded_setup", "endpoint_recorded_setup", "endpoint_cold_setup",
    }
    for trial in report["trials"].values():
        assert trial["converged"] is True
        assert trial["t_start"] == 0.0
        assert trial["dt"] == pytest.approx(0.01)
        assert trial["initial_rhs"]["all_finite"] is True
        assert trial["newton_real_matrix_condition_finite"] is True
        assert trial["newton_real_matrix_condition"] == pytest.approx(1.0)


def test_saved_step_cli_refuses_cpu_before_building_runtime(tmp_path, monkeypatch):
    path, _ = _failed_snapshot(tmp_path)
    probe = _probe_module()

    def no_gpu(*_args, **_kwargs):
        raise RuntimeError("No GPU")

    monkeypatch.setattr(probe.jax, "local_devices", no_gpu)
    with pytest.raises(SystemExit) as error:
        probe.main([str(path)])
    assert error.value.code == 2
