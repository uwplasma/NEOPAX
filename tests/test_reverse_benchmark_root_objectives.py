"""Check benchmark objective routing without VMEC, NTX, or transport solves."""

import argparse
import ast
from contextlib import nullcontext, redirect_stderr, redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from NEOPAX import _reverse_ad_optimization as optimization
from NEOPAX import _reverse_ad_parameters as parameters
from NEOPAX import _reverse_ad_transport as transport


BENCHMARK = (
    Path(__file__).resolve().parents[1]
    / "examples/benchmarks/benchmark_transport_reverse_ad_only.py"
)
MODULE = ast.parse(BENCHMARK.read_text(encoding="utf-8"))
NEW_ROOT_ROWS = (
    "Er_transition_rho", "Er_transition_location_moment", "Er_transition_strength",
)


def _namespace():
    namespace = {
        "TRANSPORT_REVERSE_OBJECTIVE_LABELS": transport.TRANSPORT_REVERSE_OBJECTIVE_LABELS,
        "TRANSPORT_OPTIMIZATION_OPTIONAL_OBJECTIVE_LABELS": (
            transport.TRANSPORT_OPTIMIZATION_OPTIONAL_OBJECTIVE_LABELS
        ),
    }
    labels = next(
        node for node in MODULE.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "FULL_TRANSPORT_OBJECTIVE_LABELS"
                for target in node.targets)
    )
    exec(compile(ast.Module(body=[labels], type_ignores=[]), str(BENCHMARK), "exec"), namespace)
    return namespace


class _EvaluationReached(Exception):
    pass


class ReverseBenchmarkRootObjectiveTests(unittest.TestCase):
    def test_all_and_individual_root_rows_reach_the_reverse_builder(self):
        """Catch a new request list paired with the builder's old default rows."""
        function = next(
            node for node in MODULE.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_run_realtime_geometry_optimization_api_smoke"
        )
        for selected in ("all", *NEW_ROOT_ROWS):
            with self.subTest(objective=selected):
                namespace = _namespace()
                labels = namespace["FULL_TRANSPORT_OBJECTIVE_LABELS"]
                self.assertEqual(len(labels), 11)
                self.assertEqual(len(set(labels)), 11)
                self.assertNotIn("Er_transition_left", labels)
                self.assertNotIn("Er_transition_right", labels)
                self.assertTrue(set(NEW_ROOT_ROWS).issubset(labels))
                self.assertIn("bootstrap_current_softmax_abs_scaled", labels)
                # Remaining modes are simply forwarded to the mocked builder.
                args = SimpleNamespace(**{
                    node.attr: None for node in ast.walk(function)
                    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id == "args"
                })
                args.full_transport_shared_payload_smoke = True
                args.optimization_api_profile_dofs = "include"
                args.objective = selected
                args.accepted_step_limit = 16
                args.reverse_segment_length = 4
                builder = Mock(return_value=object())
                evaluate = Mock(side_effect=_EvaluationReached)
                namespace.update({
                    "np": np, "jnp": np,
                    "PROFILE_PARAMETER_ORDER": parameters.PROFILE_PARAMETER_ORDER,
                    "reverse_ad_optimization_parameter_set": parameters.reverse_ad_optimization_parameter_set,
                    "VmecBoundaryParameterSpec": parameters.VmecBoundaryParameterSpec,
                    "transport_least_squares_terms": optimization.transport_least_squares_terms,
                    "LeastSquaresTerm": optimization.LeastSquaresTerm,
                    "geometry_objectives": optimization.geometry,
                    "_geometry_param_specs_from_args": lambda *_: (("RBC", 1, 0), ("ZBS", 1, 0)),
                    "_baseline_geometry_delta_vector_for_specs": lambda *_: np.zeros(2),
                    "_profile_cfg_scalar_value": lambda cfg, name: cfg[name],
                    "_make_realtime_geometry_support_segment_builder_inputs": lambda **_: (object(), object()),
                    "internal_realtime_geometry_transport_reverse_table_result_builder": builder,
                    "realtime_geometry_transport_reverse_table_request": lambda **kwargs: SimpleNamespace(**kwargs),
                    "_maybe_reverse_segment_profiler_trace": lambda _: nullcontext(),
                    "evaluate_geometry_transport_realtime_geometry_least_squares": evaluate,
                })
                extracted = ast.Module(body=[
                    ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                    function,
                ], type_ignores=[])
                exec(compile(ast.fix_missing_locations(extracted), str(BENCHMARK), "exec"), namespace)
                with redirect_stdout(io.StringIO()), self.assertRaises(_EvaluationReached):
                    namespace[function.name](
                        args=args, config={"geometry": {}, "neoclassical": {}},
                        geometry_context=object(), baseline_values=None, baseline_runtime=object(),
                        baseline_state=SimpleNamespace(pressure=np.ones((4, 5))),
                        profile_cfg={name: 1.0 for name in parameters.PROFILE_PARAMETER_ORDER},
                        neoclassical_cfg={},
                    )
                expected = labels if selected == "all" else (selected,)
                self.assertEqual(builder.call_args.kwargs["objective_labels"], labels)
                self.assertEqual(builder.call_args.kwargs["accepted_step_limit"], 16)
                self.assertEqual(builder.call_args.kwargs["reverse_segment_length"], 4)
                call = evaluate.call_args.kwargs
                self.assertEqual(call["request"].objective_names, expected)
                self.assertEqual(call["objective_labels"], labels)
                transport_terms = tuple(
                    term.objective.name for term in call["terms"]
                    if term.objective.family == "transport"
                )
                self.assertEqual(transport_terms, expected)
                self.assertIs(call["table_result_builder"], builder.return_value)

    def test_new_scalar_objectives_are_available_only_in_the_shared_transport_lane(self):
        main = next(node for node in MODULE.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        declaration = next(
            node for node in main.body
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "add_argument"
            and node.value.args and isinstance(node.value.args[0], ast.Constant)
            and node.value.args[0].value == "--objective"
        )
        guard = next(
            node for node in main.body if isinstance(node, ast.If)
            and isinstance(node.test, ast.BoolOp)
            and isinstance(node.test.values[0], ast.Compare)
            and isinstance(node.test.values[0].comparators[0], ast.Name)
            and node.test.values[0].comparators[0].id == "TRANSPORT_OPTIMIZATION_OPTIONAL_OBJECTIVE_LABELS"
        )
        namespace = _namespace()
        namespace.update({
            "OBJECTIVE_LABELS": (),
            "INITIAL_ER_ROOT_ONLY_EXPLICIT_OBJECTIVES": optimization.INITIAL_ER_ROOT_ONLY_EXPLICIT_OBJECTIVES,
            "parser": argparse.ArgumentParser(),
        })
        exec(compile(ast.Module(body=[declaration], type_ignores=[]), str(BENCHMARK), "exec"), namespace)
        for objective in NEW_ROOT_ROWS:
            with self.subTest(objective=objective):
                args = namespace["parser"].parse_args(["--objective", objective])
                args.full_transport_shared_payload_smoke = True
                args.reverse_parameter_mode = "profiles_plus_realtime_geometry"
                args.realtime_geometry_gradient_path = "reverse_payload"
                args.initial_er_root_only_optimization_smoke = False
                args.transport_solver_forward_smoke = False
                namespace["args"] = args
                check = compile(ast.Module(body=[guard], type_ignores=[]), str(BENCHMARK), "exec")
                exec(check, namespace)
                for field, invalid in (
                    ("full_transport_shared_payload_smoke", False),
                    ("reverse_parameter_mode", "profiles"),
                    ("realtime_geometry_gradient_path", "support_pullback_probe"),
                    ("initial_er_root_only_optimization_smoke", True),
                    ("transport_solver_forward_smoke", True),
                ):
                    previous = getattr(args, field)
                    setattr(args, field, invalid)
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        exec(check, namespace)
                    setattr(args, field, previous)


if __name__ == "__main__":
    unittest.main()
