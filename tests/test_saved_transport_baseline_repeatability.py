"""Saved-array diagnostics only: these tests never import JAX or run physics."""
import argparse
import ast
import contextlib
import dataclasses
import importlib.util
import json
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "examples/benchmarks/check_saved_transport_baseline_repeatability.py"
spec = importlib.util.spec_from_file_location("baseline_repeat_check", SCRIPT)
check = importlib.util.module_from_spec(spec)
_jax_before = "jax" in sys.modules
spec.loader.exec_module(check)
_jax_after = "jax" in sys.modules


class SavedBaselineTests(unittest.TestCase):
    def test_exact_comparison_catches_ulp_signed_zero_and_first_attempt(self):
        a = np.asarray([0., 1., 2., 3., 4., 5.])
        b = a.copy()
        b[5] = np.nextafter(b[5], np.inf)
        rows = check.differences({"err_norms": a}, {"err_norms": b})
        self.assertEqual(rows[0]["first_attempt"], 6)
        self.assertEqual(rows[0]["different_entries"], 1)
        self.assertEqual(rows[0]["max_abs"], b[5] - a[5])
        self.assertEqual(check.differences({"a": a}, {"a": a.copy()}), [])
        b = a.copy()
        b[0] = -0.
        row = check.differences({"a": a}, {"a": b})[0]
        self.assertEqual(row["max_abs"], 0.)
        self.assertEqual(row["first_flat_index"], 0)
        self.assertEqual(_jax_before, _jax_after)

    def test_nonfinite_dtype_shape_and_missing_fields_are_visible(self):
        a, b = np.asarray([np.inf, np.nan, 2.]), np.asarray([np.inf, np.nan, 3.])
        row = check.differences({"a": a}, {"a": b})[0]
        self.assertTrue(row["finite_masks_match"])
        self.assertEqual(row["first_flat_index"], 2)
        self.assertEqual(row["max_abs"], 1.)
        b[0] = 0.
        self.assertFalse(check.differences({"a": a}, {"a": b})[0]["finite_masks_match"])
        for other in ({}, {"a": a.astype(np.float32)}, {"a": a[:2]}):
            self.assertIn("reason", check.differences({"a": a}, other)[0])

    def test_leaf_capture_and_file_selection_preserve_data_without_calling_functions(self):
        @dataclasses.dataclass
        class Carry:
            y: object
            previous: object
            rhs: object
        def forbidden():
            self.fail("Capturing setup must not evaluate the RHS.")
        carry = Carry(np.asarray([1., 2.]), (np.asarray(3.), None), forbidden)
        leaves = check.numeric_leaves(carry, "carry")
        self.assertEqual(set(leaves), {"carry.y", "carry.previous[0]"})
        carry.y[0] = 9.
        self.assertEqual(leaves["carry.y"][0], 1.)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaises(ValueError):
                check.read_one(path, "*.npz")
            np.savez(path / "one.npz", **leaves)
            self.assertEqual(check.differences(leaves, check.read_one(path, "*.npz")), [])
            np.savez(path / "two.npz", y=np.ones(1))
            with self.assertRaises(ValueError):
                check.read_one(path, "*.npz")
            check.write_json(path / "metadata.json", {"value": np.float64(2.)})
            self.assertEqual(json.loads((path / "metadata.json").read_text()), {"value": 2.})

    def test_fresh_capture_saves_existing_setup_and_active_schedule_only(self):
        @dataclasses.dataclass
        class Carry:
            y: object
            t: object
        @dataclasses.dataclass
        class Runtime:
            geometry: object
            solver_parameters: dict
        weak = np.asarray([1., 2.]).view(type("WeakArray", (np.ndarray,), {"weak_type": True}))
        prepared = SimpleNamespace(initial_carry=Carry(weak, np.asarray(0.)))
        components = {"equation_system": SimpleNamespace(fixed_temperature_profile=np.ones((2, 3)))}
        trace = SimpleNamespace(**{name: np.arange(8.) for name in check.SCHEDULE_FIELDS})
        trace.jacobian_reused = None
        rollout = SimpleNamespace(trace=trace, final_carry=Carry(np.asarray([3., 4.]), np.asarray(0.2)),
                                  attempt_count=3, accepted_count=2)
        runtime = Runtime(np.asarray([8.]), {"rtol": 1e-6})
        with tempfile.TemporaryDirectory() as directory, patch.object(check, "version", return_value="test"):
            check.capture_fresh_baseline(directory, components=components, prepared=prepared,
                execution={"dt_min": np.asarray(1e-14)}, rollout=rollout,
                runtime=runtime, state=prepared.initial_carry, parameter_name="ZBS:1:0")
            directory = Path(directory)
            setup = check.read_one(directory, "setup.npz")
            self.assertTrue(setup["initial_carry.y.__weak_type__"])
            np.testing.assert_array_equal(setup["initial_carry.y"], [1., 2.])
            np.testing.assert_array_equal(setup["fixed_temperature_profile"], np.ones((2, 3)))
            native = check.read_one(directory, "native.npz")
            np.testing.assert_array_equal(native["err_norms"], [0., 1., 2.])
            np.testing.assert_array_equal(native["final_y"], [3., 4.])
            self.assertNotIn("jacobian_reused", native)
            metadata = json.loads((directory / "metadata.json").read_text())
            self.assertEqual(metadata["accepted_count"], 2)
            self.assertEqual(metadata["parameter_name"], "ZBS:1:0")

    def test_original_fd_baseline_only_stops_after_unchanged_schedule_call(self):
        source = SCRIPT.with_name("benchmark_transport_realtime_geometry_forward_fd.py")
        main = next(node for node in ast.parse(source.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "main")
        runtime, state, prepared, execution = (object() for _ in range(4))
        rollout = SimpleNamespace(trace=object())
        components = {"solver": object()}
        config = {"geometry": {}}
        schedule = Mock(return_value=(components, prepared, execution, rollout, object()))
        capture = Mock()
        objectives = Mock(side_effect=AssertionError("ordinary objective path reached"))
        namespace = dict(argparse=argparse, Path=Path, contextlib=contextlib,
            atexit=SimpleNamespace(register=Mock()), jnp=np,
            jax=SimpleNamespace(default_backend=lambda: "gpu", local_devices=lambda **kw: ["cuda:0"],
                                default_device=lambda device: contextlib.nullcontext()),
            DEFAULT_CONFIG=Path("unused.toml"), PARAMETER_ORDER=("n0",),
            INITIAL_ER_ROOT_ONLY_EXPLICIT_OBJECTIVES=(),
            _prepare_benchmark_config=Mock(return_value=config),
            _initial_er_root_ad_mode=lambda mode: mode,
            _baseline_profile_cfg=lambda cfg: {}, _profile_cfg_scalar_value=lambda *a: 2.,
            _frozen_linearized_vmec_geometry_bundle=Mock(return_value={}),
            _runtime_for_frozen_linearized_geometry_step=Mock(return_value=(runtime, state)),
            _profile_state_from_values=Mock(return_value=state), _schedule_rollout=schedule,
            _truncate_rollout_trace_by_accepted_steps=lambda *a: None,
            _objective_vector=objectives)
        exec(compile(ast.Module(body=[main], type_ignores=[]), str(source), "exec"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            arguments = ["fd", "--parameter", "ZBS:1:0", "--accepted-step-limit", "16",
                         "--fd-accepted-replay-contract", "reverse_frozen_metadata"]
            with patch.dict(sys.modules, {"check_saved_transport_baseline_repeatability":
                                          SimpleNamespace(capture_fresh_baseline=capture)}), \
                 patch.object(sys, "argv", arguments + ["--baseline-only-dir", directory]), \
                 contextlib.redirect_stdout(io.StringIO()):
                namespace["main"]()
            schedule.assert_called_once_with(config, runtime, state,
                                            accepted_step_limit=16, capture_newton_setup=True)
            self.assertEqual(capture.call_args.kwargs["parameter_name"], "ZBS:1:0")
            self.assertIs(capture.call_args.kwargs["prepared"], prepared)
            self.assertIs(capture.call_args.kwargs["rollout"], rollout)
            objectives.assert_not_called()
            # With no diagnostic flag, normal execution still reaches the objective path.
            capture.reset_mock()
            with patch.object(sys, "argv", arguments), contextlib.redirect_stdout(io.StringIO()), \
                 self.assertRaisesRegex(AssertionError, "ordinary objective path reached"):
                namespace["main"]()
            capture.assert_not_called()


if __name__ == "__main__":
    unittest.main()
