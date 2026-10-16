"""Opt-in scalar provenance for the first finite-to-nonfinite reverse step.

Set NEOPAX_REVERSE_STEP_DIAGNOSTICS=1 before starting a fresh process. Calls
belong outside objective vmaps: batching a conditional debug print can execute
both branches. No callback retains arrays, replays physics, or changes bars.
"""

import os

import jax
import jax.numpy as jnp


def reverse_step_diagnostics_enabled():
    # This is a trace-time switch. Existing JIT executables are not reconfigured.
    return os.environ.get("NEOPAX_REVERSE_STEP_DIAGNOSTICS", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _finite_summary(tree):
    leaves = [jnp.asarray(x) for x in jax.tree_util.tree_leaves(tree)]
    leaves = [x for x in leaves if x.size]
    finite = [jnp.isfinite(x) for x in leaves]
    all_finite = jnp.all(jnp.stack([jnp.all(mask) for mask in finite]))
    count = sum(jnp.sum(~mask) for mask in finite)
    peak = jnp.max(jnp.stack([
        jnp.max(jnp.where(mask, jnp.abs(x), 0))
        for x, mask in zip(leaves, finite)
    ]))
    return all_finite, count, peak


def reverse_step_finite_transition(stage, before, after, *, t, dt, details=None):
    """Print compact metadata only for a finite-input/nonfinite-output boundary.

    ``before`` may be a tuple for an addition boundary. The function returns
    the exact ``after`` object. It neither clips nor substitutes any values.
    Disabled calls return before converting inputs or introducing JAX effects.
    """
    if not reverse_step_diagnostics_enabled():
        return after
    before_finite, _, _ = _finite_summary(before)
    after_finite, _, _ = _finite_summary(after)

    def emit(_):
        _, _, input_peak = _finite_summary(before)
        _, count, output_peak = _finite_summary(after)
        jax.debug.print(
            "[reverse-step-first-nonfinite] stage=" + stage +
            " t={t:.16e} dt={dt:.16e} input_abs_max={input_peak:.16e} "
            "output_finite_abs_max={output_peak:.16e} nonfinite_count={count} "
            "details={details}",
            t=t, dt=dt, input_peak=input_peak, output_peak=output_peak,
            count=count, details={} if details is None else details,
            ordered=True,
        )
        return jnp.asarray(0, dtype=jnp.int32)

    jax.lax.cond(before_finite & ~after_finite, emit,
                 lambda _: jnp.asarray(0, dtype=jnp.int32), operand=None)
    return after


def reverse_step_matrix_summary(matrix):
    """Reduce an already-built stage matrix; no factorization or RHS replay."""
    finite = jnp.isfinite(matrix)
    diagonal = jnp.diag(matrix)
    first = jnp.argmax((~finite).reshape(-1))
    good = jnp.all(finite)
    return {
        "matrix_finite": good,
        "matrix_nonfinite_count": jnp.sum(~finite),
        "matrix_first_nonfinite_row": jnp.where(good, -1, first // matrix.shape[1]),
        "matrix_first_nonfinite_column": jnp.where(good, -1, first % matrix.shape[1]),
        "matrix_finite_abs_max": jnp.max(jnp.where(finite, jnp.abs(matrix), 0)),
        "matrix_diagonal_min_abs": jnp.min(jnp.abs(diagonal)),
    }


def check_reverse_segment_state(y, segment_number):
    """Stop an instrumented run before propagating a bad segment further."""
    if not reverse_step_diagnostics_enabled():
        return
    if not bool(jax.device_get(jnp.all(jnp.isfinite(y)))):
        # Flush the ordered per-step messages before the replay runner records
        # this exception. The existing segment checkpoint is already written.
        jax.effects_barrier()
        raise FloatingPointError(
            f"nonfinite transport state adjoint after reverse segment {segment_number}; "
            "NEOPAX_REVERSE_STEP_DIAGNOSTICS stopped at the first bad segment. "
            "See [reverse-step-first-nonfinite] for the per-step boundary."
        )
