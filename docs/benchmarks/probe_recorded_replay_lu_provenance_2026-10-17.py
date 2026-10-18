"""Scalar cache-fidelity audit; no transport, VMEC, NTX, or optimization.

This demonstrates incomplete Newton-setup reconstruction, not the cause of
the full-time geometry optimization regression or an incorrect adjoint.
"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from NEOPAX import _transport_solvers as solvers


def main():
    solver = solvers.RADAUSolver(
        t0=0., t1=1., dt=.001, rtol=1e-4, atol=1e-8,
        num_stages=3, maxiter=12, max_steps=8, rhs_mode="black_box",
        jacobian_reuse_mode="legacy", predictor_mode="collocation_transport_weighted",
    )
    prepared = solvers._build_prepared_radau_accepted_rollout(
        solver=solver, state=jnp.array([1.]),
        vector_field=lambda t, y, *args: -100. * y, species=None,
    )
    context = solvers._build_prepared_radau_execution_context(solver=solver, prepared_rollout=prepared)
    step = jax.jit(lambda value: solvers._radau_step_fn(context, value, None))
    initial = solvers._radau_step_state_from_carry(
        prepared.initial_carry, status=jnp.array([0, 0, 0], dtype=jnp.int32),
    )
    first, first_info = step(initial)
    rejected, reject_info = step(dataclasses.replace(first, dt=jnp.asarray(.002)))
    final, final_info = step(dataclasses.replace(rejected, dt=jnp.asarray(.001802)))
    assert bool(first_info.accepted) and not bool(reject_info.accepted) and bool(final_info.accepted)
    assert bool(final_info.jacobian_reused)
    np.testing.assert_array_equal(first.y, rejected.y)
    states = (first, final)
    schedule = (
        jnp.array([True, True]), jnp.array([.001, .001802]),
        jnp.asarray([state.dt for state in states]),
        jnp.asarray([state.recent_reject_count for state in states]),
        jnp.asarray([state.regrowth_cooldown for state in states]),
        jnp.asarray([state.easy_growth_streak for state in states]),
        jnp.asarray([state.lagged_response_valid for state in states]),
        jnp.asarray([~first_info.jacobian_reused, ~final_info.jacobian_reused]),
    )
    replay_final, replay_starts, records = solvers._radau_segment_replay_minimal_with_primal_records_call(
        context, prepared.initial_carry, schedule, validate_primal=True,
    )
    assert bool(jnp.all(records.replay_valid))
    assert not np.array_equal(final.real_lu, records.real_lu_out[-1])
    assert int(final_info.newton_iter_count) != int(records.newton_iter_count[-1])
    np.testing.assert_allclose(replay_final.y, final.y, rtol=1e-12, atol=1e-14)
    print("ODE: y'=-100*y, y(0)=1; scalar CPU only")
    print("accepted dt=0.001; rejected dt=0.002; accepted dt=0.001802")
    print("rejected error norm:", float(reject_info.err_norm))
    print("forward accepted step reused Jacobian:", bool(final_info.jacobian_reused))
    print("recorded replay valid:", np.asarray(records.replay_valid))
    print("forward / recorded replay LU:", np.asarray(final.real_lu), np.asarray(records.real_lu_out[-1]))
    print("forward / recorded replay Newton iterations:", int(final_info.newton_iter_count), int(records.newton_iter_count[-1]))
    print("absolute final-state difference:", np.abs(np.asarray(replay_final.y - final.y)))
    print("The forward reused the rejected attempt's LU; replay refactored at the accepted dt.")

    # Check the production block transpose at the converged stages. Newton
    # caches do not enter this derivative; only h and the stage RHS Jacobians do.
    physics = dataclasses.replace(context.physics_context, reverse_stage_adjoint_solve_mode="block")
    forward_start = dataclasses.replace(solvers._radau_carry_from_step_state(rejected), dt=jnp.asarray(.001802))
    forward_result = solvers._execute_radau_accepted_step_attempt_reverse_minimal(
        context.kernel_context, physics, forward_start, context.attempt_context,
        return_convergence=True,
    )
    replay_start = dataclasses.replace(
        jax.tree_util.tree_map(lambda value: value[-1], replay_starts), dt=jnp.asarray(.001802),
    )
    replay_record = jax.tree_util.tree_map(lambda value: value[-1], records)
    replay_result = solvers._radau_reverse_minimal_attempt_from_segment_primal_record(
        replay_start, context.attempt_context, replay_record,
    )

    def block_derivative(start, result):
        bars = solvers._radau_solve_exact_stage_residual_transpose_block(
            context.kernel_context, physics, start, result, None,
            rhs=result.trial_dt * context.kernel_context.b,
        )
        # For this scalar RHS, f_y=-100 and the direct endpoint y term is 1.
        return 1. + 100. * jnp.sum(bars)

    forward_derivative = block_derivative(forward_start, forward_result)
    replay_derivative = block_derivative(replay_start, replay_result)
    changed_cache = dataclasses.replace(
        replay_result, jacobian_out=jnp.full_like(replay_result.jacobian_out, 12345.),
        real_lu_out=7. * replay_result.real_lu_out,
        complex_lu_out=11. * replay_result.complex_lu_out,
    )
    changed_cache_derivative = block_derivative(replay_start, changed_cache)
    np.testing.assert_array_equal(forward_derivative, replay_derivative)
    np.testing.assert_array_equal(replay_derivative, changed_cache_derivative)
    # Independent closed-form stability function of three-stage Radau IIA.
    z = -100. * .001802
    reference = (1. + .4 * z + .05 * z ** 2) / (1. - .6 * z + .15 * z ** 2 - z ** 3 / 60.)
    np.testing.assert_allclose(replay_derivative, reference, rtol=1e-14, atol=1e-15)
    forward_stages = solvers._radau_exact_stage_times_states(context.kernel_context, forward_start, forward_result)[1]
    replay_stages = solvers._radau_exact_stage_times_states(context.kernel_context, replay_start, replay_result)[1]
    np.testing.assert_allclose(forward_stages, replay_stages, rtol=1e-12, atol=1e-14)
    print("max absolute internal-stage-state difference:", float(jnp.max(jnp.abs(forward_stages - replay_stages))))
    print("forward / replay implicit block derivative:", float(forward_derivative), float(replay_derivative))
    print("implicit derivative after changing only Newton caches:", float(changed_cache_derivative))
    print("Radau IIA stability-function reference:", reference)
    print("Different Newton iteration counts preserve the implicit derivative in this scalar check.")
    print("This does not establish full-time transport derivative parity or explain the optimization regression.")


if __name__ == "__main__":
    main()
