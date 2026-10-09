import numpy as np


def test_simple_combined_baseline_uses_ess_nominal_absolute_coordinates():
    from examples.optimization import (
        optimize_geometry_profiles_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_simple_least_squares
        as example,
    )

    args = example.parser().parse_args(())
    kwargs = example._problem_kwargs(args, physical_pitches=None)
    geometry = example.combined_example.geometry_example

    assert args.out_dir == example.OUT_DIR
    assert args.profile_dofs is True
    assert kwargs["scale_mode"] == geometry.SCALE_MODE == "ess"
    assert kwargs["ess_alpha"] == geometry.ESS_ALPHA
    assert kwargs["include_profiles"] is True
    assert kwargs["profile_parameters"] == example.combined_example.PROFILE_PARAMETERS
    assert kwargs["profile_scale_mode"] == "nominal"
    assert kwargs["profile_coordinate_mode"] == "absolute"


def test_simple_combined_baseline_adds_no_optimizer_rescaling():
    from examples.optimization import (
        optimize_geometry_profiles_qi_max_er_transition_bootstrap_net_power_initial_root_database_full_transport_simple_least_squares
        as example,
    )

    args = example.parser().parse_args(("--max-nfev", "7"))
    sentinel_evaluation = object()
    bounds = (np.asarray((-1.0,)), np.asarray((1.0,)))
    options = example._least_squares_options(
        args, sentinel_evaluation, bounds
    )

    assert options["max_nfev"] == 7
    assert options["initial_evaluation"] is sentinel_evaluation
    assert options["bounds"] is bounds
    assert "x_scale" not in options
    assert "method" not in options
