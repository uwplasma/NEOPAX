"""Benchmark callback wiring without importing its expensive runtime setup."""

import ast
import argparse
import dataclasses
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest


_ROOT = Path(__file__).resolve().parents[1]


def _function(path, name):
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return next(node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _callback(namespace):
    path = _ROOT / "examples/benchmarks/benchmark_transport_reverse_ad_only.py"
    node = _function(path, "_prepare_reverse_static_setup")
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[node.name]


def _mode_default_resolver():
    path = _ROOT / "examples/benchmarks/benchmark_transport_reverse_ad_only.py"
    node = _function(path, "_resolve_reverse_benchmark_mode_defaults")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[node.name]


def test_benchmark_callback_accepts_all_support_core_keywords():
    core = _function(
        _ROOT / "NEOPAX/_reverse_ad_transport.py",
        "prepare_realtime_geometry_support_segment_core_setup",
    )
    call = next(
        node for node in ast.walk(core)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "prepare_reverse_static_setup"
    )
    forwarded = {keyword.arg: None for keyword in call.keywords if keyword.arg is not None}
    inspect.signature(_callback({})).bind(None, **forwarded)


def _performance_args(**overrides):
    """Use the real parser declarations without executing a VMEC setup."""
    main = _function(
        _ROOT / "examples/benchmarks/benchmark_transport_reverse_ad_only.py", "main"
    )
    declarations = [
        node for node in main.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "add_argument" and node.value.args
        and isinstance(node.value.args[0], ast.Constant)
        and (
            str(node.value.args[0].value).startswith("--reverse-database-")
            or node.value.args[0].value
            in {
                "--reverse-initial-cache-support-pullback-mode",
                "--reverse-rebuild-support-pullback-mode",
            }
        )
    ]
    parser = argparse.ArgumentParser()
    exec(compile(ast.Module(body=declarations, type_ignores=[]), "<parser>", "exec"), {"parser": parser})
    args = parser.parse_args([])
    args.full_transport_shared_payload_smoke = True
    args.initial_er_root_only_optimization_smoke = False
    args.reverse_stage_adjoint_solve_mode = "block"
    args.reverse_rhs_transpose_mode = "config"
    args.reverse_stage_cotangent_mode = "full"
    args.reverse_rhs_pullback_mode = "separate"
    args.reverse_stage_adjoint_memory_mode = "default"
    args.reverse_step_bwd_mode = "reduced_cotangent_call_boundary"
    vars(args).update(overrides)
    return args


def test_validated_full_transport_reverse_modes_are_cli_defaults():
    """The measured sparse/grouped modes require no command-line overrides."""

    main = _function(
        _ROOT / "examples/benchmarks/benchmark_transport_reverse_ad_only.py", "main"
    )
    selected_flags = {
        "--reverse-bootstrap-cotangent-mode",
        "--reverse-database-interpolation-transpose-mode",
        "--reverse-database-root-interpolation-transpose-mode",
        "--reverse-database-initial-state-mode",
        "--reverse-database-initial-support-mode",
        "--reverse-database-support-preparation-mode",
        "--reverse-database-center-geometry-mode",
        "--reverse-database-stage-jacobian-mode",
        "--reverse-database-support-objective-mode",
        "--reverse-database-segment-support-mode",
        "--reverse-final-objective-cotangent-mode",
        "--reverse-initial-cache-support-pullback-mode",
        "--reverse-rebuild-support-pullback-mode",
        "--reverse-rhs-pullback-mode",
        "--reverse-rhs-transpose-mode",
        "--reverse-schedule-artifact-mode",
        "--reverse-segment-primal-record-mode",
        "--reverse-segment-start-replay-mode",
        "--reverse-stage-adjoint-memory-mode",
        "--reverse-stage-adjoint-solve-mode",
        "--reverse-stage-cotangent-mode",
        "--reverse-step-bwd-mode",
    }
    declarations = [
        node
        for node in main.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "add_argument"
        and node.value.args
        and isinstance(node.value.args[0], ast.Constant)
        and node.value.args[0].value in selected_flags
    ]
    parser = argparse.ArgumentParser()
    exec(
        compile(ast.Module(body=declarations, type_ignores=[]), "<parser>", "exec"),
        {"parser": parser},
    )
    args = parser.parse_args([])
    assert vars(args) == {
        "reverse_bootstrap_cotangent_mode": "joint_local_vjp_upar_only",
        "reverse_database_center_geometry_mode": "scalar_jvp",
        "reverse_database_initial_state_mode": "generic",
        "reverse_database_initial_support_mode": "split",
        "reverse_database_interpolation_transpose_mode": "legacy_sparse",
        "reverse_database_root_interpolation_transpose_mode": "established",
        "reverse_database_segment_support_mode": "inline",
        "reverse_database_stage_jacobian_mode": "independent",
        "reverse_database_support_objective_mode": "scalar",
        "reverse_database_support_preparation_mode": "shared",
        "reverse_final_objective_cotangent_mode": "grouped_vjp",
        "reverse_initial_cache_support_pullback_mode": "config",
        "reverse_rebuild_support_pullback_mode": "config",
        "reverse_rhs_pullback_mode": "separate",
        "reverse_rhs_transpose_mode": "config",
        "reverse_schedule_artifact_mode": "reuse_static_probe",
        "reverse_segment_primal_record_mode": "reuse_segment_primal_record",
        "reverse_segment_start_replay_mode": "minimal",
        "reverse_stage_adjoint_memory_mode": "default",
        "reverse_stage_adjoint_solve_mode": "block",
        "reverse_stage_cotangent_mode": "full",
        "reverse_step_bwd_mode": "reduced_cotangent_call_boundary",
    }


def test_database_full_transport_resolves_the_validated_support_defaults():
    args = _performance_args()
    _mode_default_resolver()(args, database_full_transport=True)
    assert args.reverse_initial_cache_support_pullback_mode == "scalar"
    assert args.reverse_rebuild_support_pullback_mode == "separate"


def test_non_database_lanes_retain_their_existing_support_defaults():
    args = _performance_args()
    _mode_default_resolver()(args, database_full_transport=False)
    assert (
        args.reverse_initial_cache_support_pullback_mode
        == "ntx_batched_interpolated_faces"
    )
    assert args.reverse_rebuild_support_pullback_mode == (
        "ntx_batched_interpolated_faces_native_multi_rhs_reuse_moment_"
        "drds_jvp_shared_primal_with_vmec_coefficients_direct_directional_"
        "product_rule"
    )


def test_explicit_support_modes_are_not_overwritten_by_default_resolution():
    args = _performance_args(
        reverse_initial_cache_support_pullback_mode="rebuild_dispatch",
        reverse_rebuild_support_pullback_mode="ntx_batched_interpolated_faces",
    )
    _mode_default_resolver()(args, database_full_transport=True)
    assert args.reverse_initial_cache_support_pullback_mode == "rebuild_dispatch"
    assert (
        args.reverse_rebuild_support_pullback_mode
        == "ntx_batched_interpolated_faces"
    )


def _check_performance_args(args, is_database=True):
    main = _function(
        _ROOT / "examples/benchmarks/benchmark_transport_reverse_ad_only.py", "main"
    )
    index = next(
        index for index, node in enumerate(main.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "database_performance_override"
                for target in node.targets)
    )
    # Real override detection, lane guard, config RHS resolution, shared-stage
    # guard, batched-support guard, deferred-segment-support guard, and sparse
    # interpolation guard.
    nodes = main.body[index:index + 8]
    assert all(isinstance(node, ast.If) for node in nodes[1:])
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<mode-checks>", "exec"), {
        "args": args, "is_database_geometry_reverse": is_database,
    })


def test_database_performance_cli_defaults_preserve_root_and_lij_lanes():
    args = _performance_args(initial_er_root_only_optimization_smoke=True,
                             full_transport_shared_payload_smoke=False)
    assert args.reverse_database_stage_jacobian_mode == "independent"
    assert args.reverse_database_support_objective_mode == "scalar"
    assert args.reverse_database_segment_support_mode == "inline"
    assert args.reverse_database_initial_support_mode == "split"
    assert args.reverse_database_initial_state_mode == "generic"
    assert args.reverse_database_support_preparation_mode == "shared"
    assert args.reverse_database_center_geometry_mode == "scalar_jvp"
    assert args.reverse_database_interpolation_transpose_mode == "legacy_sparse"
    assert (
        args.reverse_database_root_interpolation_transpose_mode
        == "established"
    )
    _check_performance_args(args, is_database=False)


@pytest.mark.parametrize("options", [
    {"initial_er_root_only_optimization_smoke": True},
    {"full_transport_shared_payload_smoke": False},
    {"reverse_stage_adjoint_solve_mode": "block_database_multi_rhs"},
    {"reverse_stage_cotangent_mode": "zero_stage_solve"},
    {"reverse_rhs_pullback_mode": "fused_ntx"},
    {"reverse_stage_adjoint_memory_mode": "stage_call_boundary"},
    {"reverse_rhs_transpose_mode": "generic"},
    {"reverse_step_bwd_mode": "current"},
])
def test_shared_stage_jacobian_cli_rejects_incompatible_contracts(options):
    with pytest.raises(SystemExit):
        _check_performance_args(_performance_args(
            reverse_database_stage_jacobian_mode="shared", **options
        ))


@pytest.mark.parametrize("stage_mode", ["shared", "shared_multi_rhs"])
def test_shared_stage_jacobian_cli_resolves_database_config_before_validation(stage_mode):
    args = _performance_args(reverse_database_stage_jacobian_mode=stage_mode)
    _check_performance_args(args)
    assert args.reverse_rhs_transpose_mode == "explicit_database"


@pytest.mark.parametrize("options", [
    {"initial_er_root_only_optimization_smoke": True},
    {"full_transport_shared_payload_smoke": False},
    {"reverse_stage_adjoint_solve_mode": "block_database_multi_rhs"},
    {"reverse_stage_cotangent_mode": "zero_stage_solve"},
    {"reverse_rhs_pullback_mode": "fused_ntx"},
    {"reverse_stage_adjoint_memory_mode": "stage_call_boundary"},
    {"reverse_rhs_transpose_mode": "generic"},
    {"reverse_step_bwd_mode": "current"},
    {"reverse_database_support_preparation_mode": "separate"},
])
def test_batched_split_support_cli_rejects_incompatible_contracts(options):
    with pytest.raises(SystemExit):
        _check_performance_args(_performance_args(
            reverse_database_support_objective_mode="batched_split", **options
        ))


def test_batched_split_support_cli_resolves_database_config_before_validation():
    args = _performance_args(
        reverse_database_support_objective_mode="batched_split"
    )
    _check_performance_args(args)
    assert args.reverse_rhs_transpose_mode == "explicit_database"


@pytest.mark.parametrize("options", [
    {"initial_er_root_only_optimization_smoke": True},
    {"full_transport_shared_payload_smoke": False},
    {"reverse_stage_adjoint_solve_mode": "block_database_multi_rhs"},
    {"reverse_stage_cotangent_mode": "zero_stage_solve"},
    {"reverse_rhs_pullback_mode": "fused_ntx"},
    {"reverse_stage_adjoint_memory_mode": "stage_call_boundary"},
    {"reverse_rhs_transpose_mode": "generic"},
    {"reverse_step_bwd_mode": "current"},
    {"reverse_database_stage_jacobian_mode": "shared"},
    {"reverse_database_support_objective_mode": "scalar"},
    {"reverse_database_support_preparation_mode": "separate"},
])
def test_deferred_segment_support_cli_rejects_incompatible_contracts(options):
    selected = {
        "reverse_database_stage_jacobian_mode": "shared_multi_rhs",
        "reverse_database_support_objective_mode": "batched_split",
        "reverse_database_segment_support_mode": "deferred_segment_batch",
    }
    selected.update(options)
    with pytest.raises(SystemExit):
        _check_performance_args(_performance_args(**selected))


def test_deferred_segment_support_cli_resolves_database_config_before_validation():
    args = _performance_args(
        reverse_database_stage_jacobian_mode="shared_multi_rhs",
        reverse_database_support_objective_mode="batched_split",
        reverse_database_segment_support_mode="deferred_segment_batch",
    )
    _check_performance_args(args)
    assert args.reverse_rhs_transpose_mode == "explicit_database"


def test_legacy_sparse_interpolation_cli_is_explicit_database_only():
    args = _performance_args(
        reverse_database_interpolation_transpose_mode="legacy_sparse"
    )
    _check_performance_args(args)
    assert args.reverse_rhs_transpose_mode == "explicit_database"
    with pytest.raises(SystemExit):
        _check_performance_args(
            _performance_args(
                reverse_database_interpolation_transpose_mode="legacy_sparse",
                reverse_rhs_transpose_mode="generic",
            )
        )


def test_selected_root_sparse_interpolation_cli_is_full_transport_only():
    args = _performance_args(
        reverse_database_root_interpolation_transpose_mode="legacy_sparse"
    )
    _check_performance_args(args)
    with pytest.raises(SystemExit):
        _check_performance_args(args, is_database=False)


def test_legacy_performance_cli_is_database_full_transport_only():
    args = _performance_args(reverse_database_initial_support_mode="generic",
                             reverse_database_support_preparation_mode="separate",
                             reverse_database_center_geometry_mode="radial_vjp")
    _check_performance_args(args)
    with pytest.raises(SystemExit):
        _check_performance_args(args, is_database=False)


def test_reduced_zero_rhs_cli_requires_the_reduced_database_contract():
    args = _performance_args(
        reverse_database_initial_support_mode="reduced_zero",
        reverse_database_initial_state_mode="reduced_zero_rhs",
    )
    _check_performance_args(args)
    with pytest.raises(SystemExit, match="reduced_zero_rhs"):
        _check_performance_args(
            _performance_args(
                reverse_database_initial_state_mode="reduced_zero_rhs",
            )
        )
    with pytest.raises(SystemExit, match="reduced_zero_rhs"):
        _check_performance_args(
            _performance_args(
                reverse_database_initial_support_mode="reduced_zero",
                reverse_database_initial_state_mode="reduced_zero_rhs",
                reverse_step_bwd_mode="current",
            )
        )


@pytest.mark.parametrize("override", [False, True])
def test_benchmark_callback_preserves_configured_execution_and_primal_contexts(override):
    @dataclasses.dataclass(frozen=True)
    class Execution:
        physics_context: object

    original_physics, selected_physics = object(), object()
    original_execution = Execution(original_physics)
    prepared = SimpleNamespace(physics_context=original_physics)
    solver = SimpleNamespace(max_steps=8)
    species, vector_field = object(), object()
    seen = []
    modes = ("reduced_zero", "reduced_zero_rhs", "separate", "radial_vjp", "shared_multi_rhs", "batched_split", "deferred_segment_batch", "legacy_sparse", "legacy_sparse") if override else (
        "split", "generic", "shared", "scalar_jvp", "independent", "scalar", "inline", "established", "established"
    )

    def _configure(physics, **kwargs):
        assert physics is original_physics
        assert kwargs.pop("vector_field") is vector_field
        assert kwargs.pop("species") is species
        seen.append(kwargs)
        return selected_physics if override else physics

    callback = _callback({
        "dataclasses": dataclasses,
        "_initial_state_for_parameter_vector": lambda *_args, **_kwargs: object(),
        "prepare_transport_solver_components": lambda *_args: {
            "solver": solver, "solve_vector_field": vector_field,
        },
        "_build_prepared_radau_accepted_rollout": lambda **_kwargs: prepared,
        "_build_prepared_radau_execution_context": lambda **_kwargs: original_execution,
        "_configure_database_reverse_performance": _configure,
        "_ReverseStaticSetup": SimpleNamespace,
    })
    options = dict(zip((
        "reverse_database_initial_support_mode",
        "reverse_database_initial_state_mode",
        "reverse_database_support_preparation_mode",
        "reverse_database_center_geometry_mode",
        "reverse_database_stage_jacobian_mode",
        "reverse_database_support_objective_mode",
        "reverse_database_segment_support_mode",
        "reverse_database_interpolation_transpose_mode",
        "reverse_database_root_interpolation_transpose_mode",
    ), modes, strict=True)) if override else {}
    result = callback(
        None, config={}, runtime=SimpleNamespace(species=species),
        baseline_state=None, profile_cfg={}, **options,
    )
    assert seen == [dict(zip((
        "initial_support_mode", "initial_state_mode", "support_preparation_mode", "center_geometry_mode",
        "stage_jacobian_mode", "support_objective_mode", "segment_support_mode",
        "interpolation_transpose_mode", "root_interpolation_transpose_mode",
    ), modes, strict=True))]
    assert result.prepared_rollout is prepared
    assert prepared.physics_context is original_physics
    if override:
        assert result.execution_context.physics_context is selected_physics
    else:
        assert result.execution_context is original_execution
