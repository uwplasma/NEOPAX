"""Check benchmark objective routing without VMEC, NTX, or transport solves."""

import argparse
import ast
from contextlib import nullcontext, redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np

from NEOPAX import _reverse_ad_optimization as optimization
from NEOPAX import _reverse_ad_parameters as parameters
from NEOPAX import _reverse_ad_transport as transport
from NEOPAX import _orchestrator as orchestrator


BENCHMARK = (
    Path(__file__).resolve().parents[1]
    / "examples/benchmarks/benchmark_transport_reverse_ad_only.py"
)
MODULE = ast.parse(BENCHMARK.read_text(encoding="utf-8"))
FD_BENCHMARK = BENCHMARK.with_name("benchmark_transport_realtime_geometry_forward_fd.py")
FD_MODULE = ast.parse(FD_BENCHMARK.read_text(encoding="utf-8"))
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
    mode_validator = next(
        node for node in MODULE.body
        if isinstance(node, ast.FunctionDef) and node.name == "_validate_replay_jacobian_reuse_mode"
    )
    exec(compile(ast.Module(body=[mode_validator], type_ignores=[]), str(BENCHMARK), "exec"), namespace)
    return namespace


def _fd_namespace():
    namespace = _namespace()
    selected = [
        node for node in FD_MODULE.body
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "OBJECTIVE_LABELS"
                    for target in node.targets)
        ) or (
            isinstance(node, ast.FunctionDef)
            and node.name in {
                "_objective_vector", "_objectives_on_realtime_geometry_frozen_trace",
                "_tree_all_finite",
            }
        )
    ]

    def terminal_value(state, runtime, index, *, objective_labels):
        if objective_labels[index] in NEW_ROOT_ROWS:
            return transport.objective_scalar_by_index(
                state, runtime, index, objective_labels=objective_labels,
            )
        # Other rows include bootstrap/NTX physics, which this test never runs.
        return jnp.asarray(index, dtype=state.Er.dtype)

    namespace.update({"jax": jax, "np": np, "jnp": jnp, "objective_scalar_by_index": terminal_value})
    module = ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *selected,
    ], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(FD_BENCHMARK), "exec"), namespace)
    return namespace


class _EvaluationReached(Exception):
    pass


class ReverseBenchmarkRootObjectiveTests(unittest.TestCase):
    def test_recorded_jacobian_reuse_requires_the_database_recorded_replay_lane(self):
        validate = _namespace()["_validate_replay_jacobian_reuse_mode"]
        self.assertEqual(validate(SimpleNamespace(), {}), "legacy")
        args = SimpleNamespace(
            reverse_replay_jacobian_reuse_mode="recorded_forward",
            full_transport_shared_payload_smoke=True,
            reverse_parameter_mode="profiles_plus_realtime_geometry",
            realtime_geometry_gradient_path="reverse_payload",
            initial_er_root_only_optimization_smoke=False,
            transport_solver_forward_smoke=False,
            reverse_segment_length=4, reverse_segment_start_replay_mode="minimal",
            reverse_segment_primal_record_mode="reuse_segment_primal_record",
            reverse_schedule_artifact_mode="reuse_static_probe",
            reverse_database_segment_support_mode="inline",
        )
        config = {"neoclassical": {"flux_model": "ntx_scan_runtime"}}
        self.assertEqual(validate(args, config), args.reverse_replay_jacobian_reuse_mode)
        for name, value in (
            ("reverse_replay_jacobian_reuse_mode", "unknown"),
            ("full_transport_shared_payload_smoke", False),
            ("reverse_parameter_mode", "profiles"),
            ("realtime_geometry_gradient_path", "support_pullback_probe"),
            ("initial_er_root_only_optimization_smoke", True),
            ("transport_solver_forward_smoke", True),
            ("reverse_segment_length", None), ("reverse_segment_length", 0),
            ("reverse_segment_start_replay_mode", "legacy"),
            ("reverse_segment_primal_record_mode", "reconstruct"),
            ("reverse_schedule_artifact_mode", "legacy"),
            ("reverse_database_segment_support_mode", "deferred_segment_batch"),
        ):
            with self.subTest(option=name, value=value):
                invalid = SimpleNamespace(**{**vars(args), name: value})
                with self.assertRaises(ValueError):
                    validate(invalid, config)
        for config in (
            {"neoclassical": {"flux_model": "ntx_exact"}},
            {"neoclassical": {"flux_model": "ntx_scan_runtime"},
             "transport_solver": {"integrator": "theta"}},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                validate(args, config)

        main = next(node for node in MODULE.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        declaration = next(
            node for node in main.body
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "add_argument"
            and node.value.args and isinstance(node.value.args[0], ast.Constant)
            and node.value.args[0].value == "--reverse-replay-jacobian-reuse-mode"
        )
        namespace = {"parser": argparse.ArgumentParser()}
        exec(compile(ast.Module(body=[declaration], type_ignores=[]), str(BENCHMARK), "exec"), namespace)
        self.assertEqual(namespace["parser"].parse_args([]).reverse_replay_jacobian_reuse_mode, "legacy")
        self.assertEqual(namespace["parser"].parse_args([
            "--reverse-replay-jacobian-reuse-mode", args.reverse_replay_jacobian_reuse_mode,
        ]).reverse_replay_jacobian_reuse_mode, args.reverse_replay_jacobian_reuse_mode)

    def test_reuse_option_does_not_switch_stages_and_reports_raw_nonfinite_values(self):
        """Only replay reuse is forwarded; root/bootstrap/geometry stages stay untouched."""
        import tempfile

        function = next(
            node for node in MODULE.body
            if isinstance(node, ast.FunctionDef) and node.name == "_run_realtime_geometry_optimization_api_smoke"
        )
        for mode, finite in (("legacy", True), ("recorded_forward", True), ("recorded_forward", False)):
            with self.subTest(mode=mode, finite=finite), tempfile.TemporaryDirectory() as directory:
                namespace = _namespace()
                labels = namespace["FULL_TRANSPORT_OBJECTIVE_LABELS"]
                args = SimpleNamespace(**{
                    node.attr: None for node in ast.walk(function)
                    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id == "args"
                })
                vars(args).update(
                    reverse_replay_jacobian_reuse_mode=mode, full_transport_shared_payload_smoke=True,
                    optimization_api_profile_dofs="include", objective="all", config="synthetic.toml",
                    reverse_parameter_mode="profiles_plus_realtime_geometry",
                    realtime_geometry_gradient_path="reverse_payload",
                    initial_er_root_only_optimization_smoke=False, transport_solver_forward_smoke=False,
                    accepted_step_limit=16, reverse_segment_length=4,
                    reverse_segment_start_replay_mode="minimal",
                    reverse_segment_primal_record_mode="reuse_segment_primal_record",
                    reverse_schedule_artifact_mode="reuse_static_probe",
                    reverse_database_segment_support_mode="inline",
                    initial_er_root_ad="jax_selected_root",
                    reverse_database_root_interpolation_transpose_mode="established",
                )
                parameter_set = parameters.reverse_ad_optimization_parameter_set(
                    include_profiles=True, profiles=parameters.PROFILE_PARAMETER_ORDER,
                    vmec_boundary=(parameters.VmecBoundaryParameterSpec("RBC", 1, 0),
                                   parameters.VmecBoundaryParameterSpec("ZBS", 1, 0)),
                )
                residual_labels = tuple(f"transport:{name}" for name in labels) + tuple(
                    f"geometry:synthetic_{i}" for i in range(7)
                )
                jacobian = np.ones((18, 8))
                if not finite:
                    jacobian[0, 0] = np.nan
                evaluation = SimpleNamespace(
                    residuals=np.ones(18), jacobian=jacobian, elapsed_s=.1,
                    result=SimpleNamespace(residual_labels=residual_labels,
                                           parameter_labels=parameter_set.labels,
                                           objective_values={name: 1. for name in labels}),
                )
                builder = Mock(return_value=object())
                evaluate = Mock(return_value=evaluation)
                root_dir = Path(directory)
                namespace.update({
                    "np": np, "jnp": np, "Path": Path, "json": json,
                    "ROOT": root_dir, "jax": SimpleNamespace(device_get=lambda value: value),
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
                    "_report_path": lambda name: root_dir / f"{name}.json",
                })
                extracted = ast.Module(body=[
                    ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), function,
                ], type_ignores=[])
                exec(compile(ast.fix_missing_locations(extracted), str(BENCHMARK), "exec"), namespace)
                runtime = SimpleNamespace(geometry=object(), species=SimpleNamespace(number_species=4))
                with redirect_stdout(io.StringIO()):
                    with nullcontext() if finite else self.assertRaises(FloatingPointError):
                        namespace[function.name](
                            args=args, config={"geometry": {}, "neoclassical": {"flux_model": "ntx_scan_runtime"}},
                            geometry_context=object(), baseline_values=None, baseline_runtime=runtime,
                            baseline_state=SimpleNamespace(pressure=np.ones((4, 5))),
                            profile_cfg={name: 1. for name in parameters.PROFILE_PARAMETER_ORDER},
                            neoclassical_cfg={},
                        )
                call = builder.call_args.kwargs
                self.assertEqual(call["accepted_step_limit"], 16)
                self.assertEqual(call["reverse_segment_length"], 4)
                self.assertEqual(call["reverse_replay_jacobian_reuse_mode"], mode)
                self.assertNotIn("reverse_stage_mode", evaluate.call_args.kwargs["options"])
                self.assertNotIn("reverse_stage_mode", evaluate.call_args.kwargs["request"].options)
                for key in ("segment_replay_optimization_stage_builder", "bootstrap_optimization_stage_builder",
                            "initial_root_optimization_stage", "runtime_optimization_stage",
                            "payload_assembly_optimization_stage"):
                    self.assertNotIn(key, call)
                report_name = "full_transport_shared_payload_smoke"
                if mode != "legacy":
                    report_name += f"_replay_jacobian_{mode}"
                report = namespace["json"].loads((root_dir / f"{report_name}.json").read_text())
                self.assertEqual(report["reverse_replay_jacobian_reuse_mode"], mode)
                self.assertEqual(report["all_finite"], finite)
                self.assertEqual(report["parameter_order"], list(parameter_set.labels))
                self.assertEqual(report["objective_order"], list(labels))
                self.assertEqual(np.asarray(report["jacobian"]).shape, (18, 8))
                if not finite:
                    self.assertTrue(np.isnan(report["jacobian"][0][0]))

    def test_geometry_context_uses_the_runtime_boozer_resolution_settings(self):
        """Omitted modes must not silently select a different FD geometry resolution."""
        function = next(
            node for node in MODULE.body
            if isinstance(node, ast.FunctionDef) and node.name == "_geometry_context_from_config"
        )
        runtime_module = ast.parse(Path(orchestrator.__file__).read_text(encoding="utf-8"))
        runtime_function = next(
            node for node in runtime_module.body
            if isinstance(node, ast.FunctionDef) and node.name == "build_runtime_context"
        )
        runtime_context_call = next(
            node for node in ast.walk(runtime_function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "build_geometry_autodiff_context"
        )
        for overrides in (
            {},
            {"mboz": 36, "nboz": 11},
            {"vmec_mboz": "24", "vmec_nboz": "7"},
            {"mboz": None, "nboz": "none"},
            {"mboz": "", "nboz": 0},
            {"mboz": 30, "nboz": 9, "vmec_mboz": 18, "vmec_nboz": 18},
        ):
            with self.subTest(settings=overrides):
                geom_cfg = {
                    "backend": "vmec_jax_booz_xform_jax",
                    "vmec_input_file": "input.synthetic", **overrides,
                }
                runtime_builder = Mock()
                eval(compile(ast.Expression(runtime_context_call), "<runtime-context>", "eval"), {
                    "geom_cfg": geom_cfg,
                    "_optional_config_int": orchestrator._optional_config_int,
                    "build_geometry_autodiff_context": runtime_builder,
                })
                benchmark_builder = Mock()
                namespace = {
                    "_parse_reverse_geometry_parameter": lambda _: ("RBC", 1, 0),
                    "_REALTIME_GEOMETRY_BACKENDS": {"vmec_jax_booz_xform_jax"},
                    "_optional_config_int": orchestrator._optional_config_int,
                    "build_geometry_autodiff_context": benchmark_builder,
                }
                extracted = ast.Module(body=[
                    ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                    function,
                ], type_ignores=[])
                exec(compile(ast.fix_missing_locations(extracted), str(BENCHMARK), "exec"), namespace)
                namespace[function.name]({"geometry": geom_cfg}, "RBC:1:0")
                for key in ("mboz", "nboz"):
                    self.assertEqual(
                        benchmark_builder.call_args.kwargs[key], runtime_builder.call_args.kwargs[key],
                    )
                if not overrides:
                    self.assertIsNone(benchmark_builder.call_args.kwargs["mboz"])
                    self.assertIsNone(benchmark_builder.call_args.kwargs["nboz"])

    def test_fd_objective_order_and_root_endpoint_differences_match_terminal_ad(self):
        namespace = _fd_namespace()
        labels = namespace["OBJECTIVE_LABELS"]
        self.assertEqual(labels, _namespace()["FULL_TRANSPORT_OBJECTIVE_LABELS"])
        rho = jnp.linspace(0.003, 0.997, 51)
        er = 40 * jnp.tanh((0.56 - rho) / 0.018)
        direction = 0.3 + jnp.sin(4 * rho)
        runtime = SimpleNamespace(geometry=SimpleNamespace(rho_grid=rho))

        def objective_values(delta):
            return namespace["_objective_vector"](
                SimpleNamespace(Er=er + delta * direction), runtime,
            )

        values = np.asarray(objective_values(0.0))
        self.assertEqual(values.shape, (11,))
        self.assertTrue(np.isfinite(values).all())
        indices = [labels.index(name) for name in NEW_ROOT_ROWS]
        radius, moment, strength = values[indices]
        self.assertAlmostEqual(moment, radius - 0.5, places=12)
        self.assertGreater(strength, 0.0)
        step = 1e-4
        fd = np.asarray((objective_values(step) - objective_values(-step)) / (2 * step))
        ad = np.asarray(jax.jacfwd(objective_values)(0.0))
        np.testing.assert_allclose(fd[indices], ad[indices], rtol=1e-5, atol=1e-8)
        self.assertTrue((np.abs(ad[indices]) > 1e-8).all())

    def test_fd_replay_endpoints_return_all_new_objectives(self):
        namespace = _fd_namespace()
        rho = jnp.linspace(0.003, 0.997, 51)
        state = SimpleNamespace(Er=40 * jnp.tanh((0.56 - rho) / 0.018))
        runtime = SimpleNamespace(geometry=SimpleNamespace(rho_grid=rho))
        prepared = SimpleNamespace(
            initial_carry=object(),
            physics_context=SimpleNamespace(unpack_flat=lambda _: state),
        )
        frozen_replay = SimpleNamespace(final_carry=SimpleNamespace(y=object()))
        namespace.update({
            "_profile_state_from_values": lambda *_args, **_kwargs: state,
            "_prepare_rollout": lambda *_args, **_kwargs: (None, prepared, object()),
            "_radau_forward_fd_run_prepared_on_realized_trace": Mock(return_value={"final_state": state}),
            "_radau_run_prepared_on_realized_trace": Mock(return_value={"final_state": state}),
            "_fd_reverse_frozen_metadata_accepted_rollout": Mock(return_value=frozen_replay),
            "_require_complete_frozen_metadata_replay": Mock(),
        })
        expected = namespace["_objective_vector"](state, runtime)
        for mode, contract in (
            ("accepted", "current"),
            ("accepted", "reverse_frozen_metadata"),
            ("attempt", "current"),
        ):
            with self.subTest(mode=mode, contract=contract):
                values, replay = namespace["_objectives_on_realtime_geometry_frozen_trace"](
                    config={}, runtime=runtime, baseline_state=state, profile_cfg={},
                    profile_values=None, frozen_trace=object(), replay_mode=mode,
                    fd_accepted_replay_contract=contract, baseline_replay_reference=object(),
                )
                self.assertIs(replay["final_state"], state)
                np.testing.assert_array_equal(values, expected)

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
