import ast
from pathlib import Path

import numpy as np


def test_simple_combined_baseline_uses_ess_nominal_absolute_coordinates():
    from examples.optimization import (
        optimize_geometry_profiles_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_simple_least_squares
        as example,
    )

    args = example.parser().parse_args(())
    kwargs = example._problem_kwargs(args, physical_pitches=None)
    geometry = example.geometry_example

    assert args.out_dir == example.OUT_DIR
    assert kwargs["scale_mode"] == geometry.SCALE_MODE == "ess"
    assert kwargs["ess_alpha"] == geometry.ESS_ALPHA
    assert kwargs["include_profiles"] is True
    assert kwargs["profile_parameters"] == example.PROFILE_PARAMETERS
    assert kwargs["profile_scale_mode"] == "nominal"
    assert kwargs["profile_coordinate_mode"] == "absolute"


def test_simple_combined_baseline_is_geometry_problem_plus_profiles_only():
    from examples.optimization import (
        optimize_geometry_profiles_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_simple_least_squares
        as example,
    )

    root = Path(__file__).resolve().parents[1]
    geometry_tree = ast.parse(
        (root / "examples/optimization/optimize_geometry_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport.py").read_text(
            encoding="utf-8"
        )
    )
    geometry_call = next(
        node
        for node in ast.walk(geometry_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "geometry_full_transport_least_squares_problem"
    )
    geometry_keywords = {keyword.arg for keyword in geometry_call.keywords}

    args = example.parser().parse_args(())
    combined_keywords = set(
        example._problem_kwargs(args, physical_pitches=None)
    )
    assert combined_keywords - geometry_keywords == {
        "include_profiles",
        "profile_parameters",
        "profile_scale_mode",
        "profile_coordinate_mode",
    }
    assert geometry_keywords - combined_keywords == set()


def test_simple_combined_baseline_preserves_geometry_import_prelude():
    from examples.optimization import (
        optimize_geometry_profiles_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_simple_least_squares
        as example,
    )

    source = Path(example.__file__).read_text(encoding="utf-8")
    assert source.index("import jax\n") < source.index("import jax.numpy as jnp")
    assert source.index("import jax.numpy as jnp") < source.index("import vmex as vj")
    assert source.index("import vmex as vj") < source.index(
        "from NEOPAX import optimization as opt"
    )


def test_simple_combined_profile_bounds_are_physical_nominal_bounds():
    from examples.optimization import (
        optimize_geometry_profiles_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_simple_least_squares
        as example,
    )

    assert example.PROFILE_PHYSICAL_LOWER["n0"] == 0.6
    assert example.PROFILE_PHYSICAL_UPPER["n0"] == 10.0
    assert example.PROFILE_PHYSICAL_LOWER["T0"] == 5.0
    assert example.PROFILE_PHYSICAL_UPPER["T0"] == 25.0


def test_simple_combined_postprocess_runs_both_fresh_transport_reports():
    from examples.optimization import (
        optimize_geometry_profiles_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_simple_least_squares
        as example,
    )

    tree = ast.parse(Path(example.__file__).read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "write_outputs"
    )
    calls = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "geometry_example"
        and node.func.attr == "write_transport_report"
    ]
    assert len(calls) == 2
    assert {
        ast.literal_eval(call.args[1])
        for call in calls
    } == {"initial", "optimized"}


def test_simple_combined_postprocess_restores_forward_progress_flags():
    from examples.optimization import (
        optimize_geometry_profiles_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_simple_least_squares
        as example,
    )

    base = {
        "general": {"device": "default"},
        "profiles": {"n0": 4.21, "T0": 17.8},
        "transport_solver": {
            "debug_stage_markers": True,
            "debug_walltime_attempts": True,
        },
    }
    result = example.postprocess_config_with_profiles(
        base, {"n0": 3.5, "T0": 20.0}
    )

    assert result["profiles"] == {"n0": 3.5, "T0": 20.0}
    assert result["transport_solver"]["debug_stage_markers"] is True
    assert result["transport_solver"]["debug_walltime_attempts"] is True
    assert result["general"]["device"] == "auto"
    assert base["profiles"] == {"n0": 4.21, "T0": 17.8}


def test_simple_combined_profile_trust_multiplier_leaves_geometry_at_unit_scale():
    from examples.optimization import (
        optimize_geometry_profiles_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_simple_least_squares
        as example,
    )

    class Problem:
        parameter_labels = (
            "n0",
            "T0",
            "RBC:1:0",
            "density_shape_alpha",
            "ZBS:1:0",
        )

    np.testing.assert_array_equal(
        example.optimizer_trust_region_x_scale(Problem(), 3.7),
        np.asarray([3.7, 3.7, 1.0, 3.7, 1.0]),
    )
    np.testing.assert_array_equal(
        example.optimizer_trust_region_x_scale(Problem(), 1.0),
        np.ones(5),
    )


def test_simple_combined_passes_fixed_profile_block_trust_scale_to_scipy():
    from examples.optimization import (
        optimize_geometry_profiles_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_simple_least_squares
        as example,
    )

    args = example.parser().parse_args(("--max-nfev", "7"))
    assert args.profile_trust_multiplier == example.PROFILE_TRUST_MULTIPLIER == 3.7
    sentinel_evaluation = object()
    bounds = (np.asarray((-1.0,)), np.asarray((1.0,)))
    x_scale = np.asarray((3.7, 1.0))
    options = example._least_squares_options(
        args, sentinel_evaluation, bounds, x_scale
    )

    assert options["max_nfev"] == 7
    assert options["initial_evaluation"] is sentinel_evaluation
    assert options["bounds"] is bounds
    np.testing.assert_array_equal(options["x_scale"], x_scale)
    assert "method" not in options
