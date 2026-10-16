"""Small provenance/2x2 linear-solve checks; no transport or geometry solve."""

from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from NEOPAX import _transport_solvers as solvers
from NEOPAX._reverse_step_diagnostics import (
    check_reverse_segment_state,
    reverse_step_finite_transition,
)


class ReverseStepDiagnosticsTests(unittest.TestCase):
    def test_disabled_does_not_convert_inputs(self):
        sentinel = object()
        with patch.dict(os.environ, {"NEOPAX_REVERSE_STEP_DIAGNOSTICS": "0"}):
            self.assertIs(reverse_step_finite_transition(
                "disabled", sentinel, sentinel, t=0., dt=1.), sentinel)

    def test_scan_reports_transition_once_and_preserves_nonfinite_carry(self):
        output = io.StringIO()
        with patch.dict(os.environ, {"NEOPAX_REVERSE_STEP_DIAGNOSTICS": "1"}), redirect_stdout(output):
            @jax.jit
            def run():
                def step(value, index):
                    result = jnp.where(index == 1, jnp.full_like(value, jnp.nan), value * 2)
                    returned = reverse_step_finite_transition(
                        "scan_test", value, result, t=index, dt=1.)
                    return returned, None
                return jax.lax.scan(step, jnp.ones((3, 2)), jnp.arange(4))[0]

            result = np.asarray(run())
            jax.effects_barrier()
        self.assertTrue(np.all(np.isnan(result)))
        text = output.getvalue()
        self.assertEqual(text.count("[reverse-step-first-nonfinite]"), 1, text)
        self.assertIn("stage=scan_test", text)
        self.assertIn("input_abs_max=2.", text)

    def test_addition_overflow_reports_only_with_finite_operands(self):
        output = io.StringIO()
        with patch.dict(os.environ, {"NEOPAX_REVERSE_STEP_DIAGNOSTICS": "1"}), redirect_stdout(output):
            @jax.jit
            def run(a, b):
                summed = a + b
                return reverse_step_finite_transition(
                    "addition_test", (a, b), summed, t=0., dt=1.)

            maximum = np.finfo(np.float64).max
            overflow = np.asarray(run(jnp.asarray([maximum]), jnp.asarray([maximum])))
            propagated = np.asarray(run(jnp.asarray([np.inf]), jnp.asarray([1.])))
            jax.effects_barrier()
        self.assertTrue(np.isposinf(overflow[0]))
        self.assertTrue(np.isposinf(propagated[0]))
        self.assertEqual(output.getvalue().count("[reverse-step-first-nonfinite]"), 1)

    def test_actual_batched_block_solve_reports_matrix_without_extra_assembly(self):
        kernel = SimpleNamespace(dtype=jnp.float64, num_stages=1, state_dim=2)
        physics = SimpleNamespace(reverse_stage_adjoint_solve_mode="block")
        carry = SimpleNamespace(t=jnp.asarray(0.25))
        primal = SimpleNamespace(trial_dt=jnp.asarray(0.1))
        rhs = jnp.asarray([[1., 2.], [0., 0.], [-3., 4.]])
        matrices = (
            np.array([[2., 0.], [0., 4.]]),
            np.array([[1., 0.], [0., 0.]]),
            np.array([[np.nan, 0.], [0., 1.]]),
        )
        for matrix in matrices:
            with self.subTest(matrix=matrix):
                output = io.StringIO()
                def build_matrix(*args, return_diagnostics=False):
                    value = jnp.asarray(matrix)
                    if return_diagnostics:
                        return value, {}, {"stage_states": jnp.ones((1, 2))}
                    return value

                with patch.dict(os.environ, {"NEOPAX_REVERSE_STEP_DIAGNOSTICS": "1"}), \
                        patch.object(solvers, "_radau_exact_stage_residual_matrix", side_effect=build_matrix) as assembly, \
                        redirect_stdout(output):
                    run = jax.jit(lambda value: solvers._radau_solve_exact_stage_residual_transpose_batched(
                        kernel, physics, carry, primal, None, rhs=value))
                    result = np.asarray(run(rhs))
                    jax.effects_barrier()
                self.assertEqual(assembly.call_count, 1)
                text = output.getvalue()
                if np.array_equal(matrix, matrices[0]):
                    np.testing.assert_allclose(result, np.linalg.solve(matrix.T, -np.asarray(rhs).T).T)
                    self.assertEqual(text, "")
                else:
                    self.assertFalse(np.all(np.isfinite(result)))
                    self.assertEqual(text.count("[reverse-step-first-nonfinite]"), 1, text)
                    self.assertIn("stage=stage_adjoint_dense_solve", text)
                    self.assertIn("matrix_finite", text)
                    self.assertIn("matrix_first_nonfinite_row", text)

    def test_primal_aux_separates_bad_replay_values_from_bad_derivatives(self):
        kernel = SimpleNamespace(dtype=jnp.float64, num_stages=1, state_dim=2,
                                 a=jnp.array([[.5]]), c=jnp.array([.5]))
        carry = SimpleNamespace(t=jnp.asarray(.25), y=jnp.array([1., 2.]))

        @jax.custom_jvp
        def bad_derivative(y):
            return -(y ** 2)

        @bad_derivative.defjvp
        def bad_derivative_jvp(primals, tangents):
            (y,), (dy,) = primals, tangents
            return -(y ** 2), jnp.full_like(dy, jnp.nan)

        for kind in ("finite", "bad_derivative", "bad_primal"):
            with self.subTest(kind=kind):
                calls = []

                def rhs(t, y):
                    calls.append(True)
                    return bad_derivative(y) if kind == "bad_derivative" else -(y ** 2)

                physics = SimpleNamespace(reverse_stage_adjoint_solve_mode="block",
                                          flat_rhs=rhs, flat_rhs_with_lagged_response=None)
                slopes = jnp.array([[-1., -2.]])
                if kind == "bad_primal":
                    slopes = jnp.full_like(slopes, jnp.nan)
                primal = SimpleNamespace(trial_dt=jnp.asarray(.1), trial_y=jnp.array([.9, 1.8]),
                                         stage_history=slopes)
                run = jax.jit(lambda: solvers._radau_exact_stage_residual_matrix(
                    kernel, physics, carry, primal, None, return_diagnostics=True))
                matrix, details, snapshot = jax.device_get(run())
                # The primal RHS is aux from that same jacfwd, not a second
                # function call to get diagnostic values.
                self.assertEqual(len(calls), 1)
                self.assertTrue(details["step_input_y_finite"])
                self.assertTrue(details["trial_y_finite"])
                self.assertTrue(details["stage_rhs_captured"])
                self.assertEqual(bool(details["stage_states_finite"][0]), kind != "bad_primal")
                self.assertEqual(bool(details["stage_rhs_finite"][0]), kind != "bad_primal")
                self.assertEqual(bool(details["stage_jacobians_finite"][0]), kind == "finite")
                if kind == "finite":
                    states = np.asarray(carry.y)[None, :] + .05 * np.asarray(slopes)
                    np.testing.assert_allclose(matrix, np.diag(1 + .1 * states[0]))
                    np.testing.assert_allclose(snapshot["stage_rhs"], -(states ** 2))
                else:
                    self.assertFalse(np.all(np.isfinite(matrix)))

    def test_only_transition_saves_the_small_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with patch.dict(os.environ, {"NEOPAX_REVERSE_STEP_DIAGNOSTICS": "1",
                                        "NEOPAX_REVERSE_STEP_SNAPSHOT_DIR": directory}), redirect_stdout(output):
                @jax.jit
                def run(before, after):
                    return reverse_step_finite_transition(
                        "snapshot_test", before, after, t=.25, dt=.1,
                        snapshot={"stage_states": before[None, :]})

                good = jnp.array([1., 2.])
                bad = jnp.array([jnp.nan, jnp.nan])
                run(good, good).block_until_ready()
                run(good, bad).block_until_ready()
                run(bad, bad).block_until_ready()
                jax.effects_barrier()
            files = list(Path(directory).glob("*.npz"))
            self.assertEqual(len(files), 1)
            self.assertEqual(output.getvalue().count("[reverse-step-snapshot] wrote"), 1)
            with np.load(files[0]) as data:
                np.testing.assert_array_equal(data["stage_states"], [[1., 2.]])
                self.assertEqual(float(data["t"]), .25)
                self.assertEqual(str(data["diagnostic_stage"]), "snapshot_test")

    def test_segment_guard_is_opt_in_and_preserves_finite_values(self):
        with patch.dict(os.environ, {"NEOPAX_REVERSE_STEP_DIAGNOSTICS": "0"}):
            check_reverse_segment_state(object(), 5)
        with patch.dict(os.environ, {"NEOPAX_REVERSE_STEP_DIAGNOSTICS": "1"}):
            check_reverse_segment_state(jnp.array([[1., -35.]]), 5)
            with self.assertRaisesRegex(FloatingPointError, "reverse segment 5"):
                check_reverse_segment_state(jnp.array([[1., jnp.nan]]), 5)


if __name__ == "__main__":
    unittest.main()
