"""Tiny scalar probe of accepted-only Newton cache reconstruction; no plasma physics."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from NEOPAX import _transport_solvers as solvers


def main():
    solver = solvers.RADAUSolver(
        t0=0., t1=1., dt=.001, rtol=.01, atol=1e-8,
        num_stages=3, maxiter=12, max_steps=8, rhs_mode="black_box",
        jacobian_reuse_mode="legacy", predictor_mode="collocation_transport_weighted",
    )
    prepared = solvers._build_prepared_radau_accepted_rollout(
        solver=solver, state=jnp.array([1.]),
        vector_field=lambda t, y, *args: -100. * y ** 3, species=None,
    )
    context = solvers._build_prepared_radau_execution_context(solver=solver, prepared_rollout=prepared)
    step = jax.jit(lambda value: solvers._radau_step_fn(context, value, None))
    state = solvers._radau_step_state_from_carry(
        prepared.initial_carry, status=jnp.array([0, 0, 0], dtype=jnp.int32),
    )
    state, first = step(state)
    assert bool(first.accepted), "Probe requires its first small step to be accepted."
    checkpoint = solvers._radau_carry_from_step_state(state)
    rejected_state, rejected = step(dataclasses.replace(state, dt=jnp.asarray(.05)))
    assert not bool(rejected.accepted), "Probe requires its large trial to be rejected."
    accepted_state, last = step(dataclasses.replace(rejected_state, dt=jnp.asarray(.001)))
    assert bool(last.accepted), "Probe requires its final small step to be accepted."
    replay = jax.jit(lambda carry: solvers._radau_checked_primal_replay_record(context, carry))(
        dataclasses.replace(checkpoint, dt=jnp.asarray(.001)),
    )
    original = solvers._execute_radau_accepted_step_attempt_reverse_minimal(
        context.kernel_context, context.physics_context,
        dataclasses.replace(solvers._radau_carry_from_step_state(rejected_state), dt=jnp.asarray(.001)),
        context.attempt_context, return_convergence=True,
    )
    assert np.array_equal(checkpoint.y, rejected_state.y)
    print("ODE: y' = -100*y**3; identical physical input y/t/dt for the compared accepted step")
    print("kernel Jacobian reuse mode:", context.kernel_context.jacobian_reuse_mode)
    print("first accepted / large rejected / final accepted:", bool(first.accepted), bool(rejected.accepted), bool(last.accepted))
    print("first accepted Newton iterations:", int(first.newton_iter_count))
    print("checkpoint state:", np.asarray(checkpoint.y))
    print("forward Jacobian used:", np.asarray(original.jacobian_out))
    print("accepted-only replay Jacobian used:", np.asarray(replay.jacobian_out))
    print("forward / replay Newton iterations:", int(original.newton_iter_count), int(replay.newton_iter_count))
    print("forward / replay Newton converged:", bool(original.converged), bool(replay.replay_converged))
    print("forward / replay final state:", np.asarray(accepted_state.y), np.asarray(replay.trial_y))
    print("absolute state difference:", np.abs(np.asarray(accepted_state.y) - np.asarray(replay.trial_y)))
    schedule = (
        jnp.array([True, True]), jnp.array([.001, .001]), jnp.array([.001, .001]),
        jnp.zeros(2, dtype=jnp.int32), jnp.zeros(2, dtype=jnp.int32),
        jnp.zeros(2, dtype=jnp.int32), jnp.zeros(2, dtype=bool),
        jnp.array([~first.jacobian_reused, ~last.jacobian_reused]),
    )
    fixed, _, records = solvers._radau_segment_replay_minimal_with_primal_records_call(
        context, prepared.initial_carry, schedule, validate_primal=True,
    )
    np.testing.assert_allclose(records.jacobian_out[-1], original.jacobian_out, rtol=1e-13)
    np.testing.assert_allclose(fixed.y, accepted_state.y, rtol=1e-13)
    assert int(records.newton_iter_count[-1]) == int(original.newton_iter_count)
    print("recorded-decision replay Jacobian / iterations:", np.asarray(records.jacobian_out[-1]), int(records.newton_iter_count[-1]))
    print("recorded-decision replay absolute state difference:", np.abs(np.asarray(fixed.y) - np.asarray(accepted_state.y)))
    print("This probes a cache mismatch, not the cause of the remote transport NaNs.")


if __name__ == "__main__":
    main()
