"""Opt-in scalar provenance for the first finite-to-nonfinite reverse step.

Set NEOPAX_REVERSE_STEP_DIAGNOSTICS=1 before starting a fresh process. Calls
belong outside objective vmaps: batching a conditional debug print can execute
both branches. NEOPAX_REVERSE_STEP_SNAPSHOT_DIR optionally saves the failing
stage inputs. No callback retains device history, replays physics, or changes
bars.
"""

import os
from pathlib import Path
import tempfile

import jax
import jax.numpy as jnp
import numpy as np


def reverse_step_diagnostics_enabled():
    # This is a trace-time switch. Existing JIT executables are not reconfigured.
    return os.environ.get("NEOPAX_REVERSE_STEP_DIAGNOSTICS", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def save_reverse_step_snapshot(stage, values, time_value, dt_value, *, directory=None):
    """Save one small host snapshot when its directory was explicitly enabled."""
    directory = directory or os.environ.get("NEOPAX_REVERSE_STEP_SNAPSHOT_DIR", "")
    if not directory:
        return
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=directory, prefix="bad_reverse_step_", suffix=".npz", delete=False,
    ) as stream:
        np.savez_compressed(
            stream, **{name: np.asarray(value) for name, value in values.items()},
            t=np.asarray(time_value), dt=np.asarray(dt_value), diagnostic_stage=np.asarray(stage),
        )
        filename = stream.name
    print(f"[reverse-step-snapshot] wrote {filename}", flush=True)


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


def reverse_step_finite_transition(stage, before, after, *, t, dt, details=None, snapshot=None):
    """Print compact metadata only for a finite-input/nonfinite-output boundary.

    ``before`` may be a tuple for an addition boundary. The function returns
    the exact ``after`` object. It neither clips nor substitutes any values.
    Disabled calls return before converting inputs or introducing JAX effects.
    A supplied snapshot is saved only at a transition and only when its output
    directory is explicitly enabled through the environment.
    """
    if not reverse_step_diagnostics_enabled():
        return after
    snapshot_dir = os.environ.get("NEOPAX_REVERSE_STEP_SNAPSHOT_DIR", "")
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
        if snapshot is not None and snapshot_dir:
            def save_snapshot(values, time_value, dt_value):
                save_reverse_step_snapshot(stage, values, time_value, dt_value, directory=snapshot_dir)

            jax.debug.callback(save_snapshot, snapshot, t, dt, ordered=True)
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


def check_primal_replay_endpoint(actual, expected, *, atol, rtol):
    """Require a checked replay to match its adaptive endpoint within state tolerances."""
    actual, expected = jnp.asarray(actual), jnp.asarray(expected)
    difference = jnp.abs(actual - expected)
    scale = jnp.asarray(atol) + jnp.asarray(rtol) * jnp.maximum(jnp.abs(actual), jnp.abs(expected))
    peak, scaled_peak = jax.device_get((
        jnp.max(difference),
        jnp.max(difference / jnp.maximum(scale, jnp.finfo(actual.dtype).tiny)),
    ))
    if not np.isfinite(scaled_peak) or scaled_peak > 1.0:
        raise FloatingPointError(
            f"primal replay endpoint differs from adaptive checkpoint: "
            f"max_abs_difference={float(peak):.6e} max_scaled_difference={float(scaled_peak):.6e}; "
            "transport adjoint was not evaluated for this segment."
        )
    return float(peak), float(scaled_peak)
