import ast
from pathlib import Path

import numpy as np


def _example():
    from examples.optimization import (
        optimize_geometry_warm_start_profiles_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport
        as example,
    )

    return example


def test_warm_start_parser_uses_two_full_budgets_and_unit_profile_trust():
    example = _example()
    args = example.parser().parse_args(())

    assert args.geometry_max_nfev is None
    assert args.combined_max_nfev is None
    assert example._phase_budget(args.geometry_max_nfev, args.max_nfev) == 30
    assert example._phase_budget(args.combined_max_nfev, args.max_nfev) == 30
    assert args.profile_coordinate_mode == "delta"
    assert args.profile_trust_multiplier == 1.0


def test_warm_start_geometry_phase_has_exact_geometry_only_keyword_bundle():
    example = _example()
    root = Path(__file__).resolve().parents[1]
    geometry_source = root / (
        "examples/optimization/"
        "optimize_geometry_qi_max_er_transition_bootstrap_net_power_initial_root_"
        "database_full_transport.py"
    )
    tree = ast.parse(geometry_source.read_text(encoding="utf-8"))
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "geometry_full_transport_least_squares_problem"
    )
    established_keywords = {keyword.arg for keyword in call.keywords}

    args = example.parser().parse_args(())
    warm_keywords = set(example._geometry_problem_kwargs(args, None))
    warm_keywords.add("vmec_input")

    assert warm_keywords == established_keywords
    assert "include_profiles" not in warm_keywords
    assert "profile_parameters" not in warm_keywords


def test_warm_start_imports_combined_helper_only_after_geometry_phase():
    example = _example()
    source = Path(example.__file__).read_text(encoding="utf-8")

    geometry_call = source.index("geometry_phase = _run_geometry_phase(")
    combined_import = source.index("as combined_helper,")
    assert geometry_call < combined_import


def test_warm_start_uses_nonlinear_least_squares_in_both_phases():
    example = _example()
    tree = ast.parse(Path(example.__file__).read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "opt"
        and node.func.attr == "least_squares"
    ]

    assert len(calls) == 2


def test_warm_start_appends_profiles_to_accepted_geometry_coordinates():
    example = _example()

    class GeometryProblem:
        parameter_labels = ("RBC:1:0", "ZBS:1:0")
        x_scale = np.asarray((0.25, 0.5))

    class GeometryResult:
        x = np.asarray((0.125, -0.375))

    class CombinedProblem:
        parameter_labels = ("n0", "T0", "RBC:1:0", "ZBS:1:0")
        x_scale = np.asarray((4.21, 17.8, 0.25, 0.5))
        x0 = np.zeros(4)

    np.testing.assert_array_equal(
        example._combined_starting_point(
            GeometryProblem(), GeometryResult(), CombinedProblem()
        ),
        np.asarray((0.0, 0.0, 0.125, -0.375)),
    )


def test_warm_start_rejects_recomputed_geometry_scale():
    example = _example()

    class GeometryProblem:
        parameter_labels = ("RBC:1:0",)
        x_scale = np.asarray((0.25,))

    class GeometryResult:
        x = np.asarray((0.125,))

    class CombinedProblem:
        parameter_labels = ("n0", "RBC:1:0")
        x_scale = np.asarray((4.21, 0.5))
        x0 = np.zeros(2)

    try:
        example._combined_starting_point(
            GeometryProblem(), GeometryResult(), CombinedProblem()
        )
    except AssertionError as exc:
        assert "changed the phase-1 ESS geometry scales" in str(exc)
    else:
        raise AssertionError("A recomputed ESS scale must be rejected.")


def test_warm_start_does_not_modify_geometry_only_source():
    example = _example()
    source = Path(example.__file__).read_text(encoding="utf-8")

    assert "geometry_example.main(" not in source
    assert "subprocess" not in source
    assert "include_profiles" not in source[
        source.index("def _build_geometry_problem(") : source.index("def _cost(")
    ]
    assert 'current_input = geometry_phase["baseline_input"]' in source
    assert 'current_input = geometry_phase["warm_path"]' not in source
