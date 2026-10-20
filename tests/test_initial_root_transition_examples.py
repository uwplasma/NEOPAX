"""Root-only example wiring checks without VMEC, NTX, roots, or transport solves."""

import copy
import importlib
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from NEOPAX import optimization as opt
from NEOPAX import _reverse_ad_optimization as reverse


EXAMPLES = [
    "optimize_geometry_qi_max_er_transition_initial_root",
    "optimize_geometry_qi_max_er_transition_initial_root_database",
    "optimize_geometry_qi_max_er_transition_initial_root_finitebeta",
    "optimize_geometry_qi_max_er_transition_initial_root_database_finitebeta",
    "optimize_geometry_qi_max_er_transition_bootstrap_initial_root",
    "optimize_geometry_qi_max_er_transition_bootstrap_initial_root_database",
    "optimize_geometry_qi_max_er_transition_bootstrap_initial_root_finitebeta",
    "optimize_geometry_qi_max_er_transition_bootstrap_initial_root_database_finitebeta",
    "optimize_profiles_max_er_transition_bootstrap_initial_root",
]


@pytest.mark.parametrize("name", EXAMPLES)
def test_root_only_examples_assemble_new_residuals_and_report_costs(name):
    module = importlib.import_module(f"examples.optimization.{name}")
    terms = getattr(module, "TERMS", getattr(module, "terms", None))
    if terms is None:
        terms = module.active_terms()
    normalized = tuple(
        term for term in opt._normalize_initial_er_root_least_squares_terms(terms)
        if term.objective.family == "transport" and term.objective.name.startswith("Er_transition_")
    )
    assert tuple(term.objective.name for term in normalized) == (
        "Er_transition_rho", "Er_transition_location_moment",
        "Er_transition_strength", "Er_transition_strength",
    )
    reverse.normalize_initial_er_root_only_objective_names(tuple(term.objective.name for term in normalized))
    assert "electron_root" in module.OUT_DIR.name
    assert module.ROOT_OPTIONS["Er_transition_rho_target"] == 0.514
    assert not any("index" in key for key in module.ROOT_OPTIONS)

    # Raw values have independent parameter directions; validate the actual
    # mixed-term chain rule, including satisfied and unsatisfied strength.
    matrix = np.asarray([[0.03, -0.04], [0.03, -0.04], [0.4, 0.6], [0.4, 0.6]])
    for strength in (-2.0, 0.9, 2.0):
        raw = np.asarray([0.6, 0.6 - module.ER_TRANSITION_RHO_TARGET, strength, strength])
        base = reverse.LeastSquaresResult(
            residuals=jnp.asarray(raw), jacobian=jnp.asarray(matrix),
            residual_labels=tuple(term.objective.label for term in normalized),
            parameter_labels=("profile", "geometry"), objective_values={},
        )
        evaluation = reverse.LeastSquaresEvaluation(
            result=base, residuals=base.residuals, jacobian=base.jacobian, elapsed_s=0.0,
        )
        actual = opt._assemble_mixed_initial_er_root_result(normalized, base_evaluation=evaluation)

        def independent_residual(x):
            values = raw + matrix @ x
            deficit = module.ER_TRANSITION_STRENGTH_TARGET - values[3]
            smooth_deficit = 0.5 * (deficit + np.hypot(deficit, 1e-6))
            return np.asarray([
                0.0, np.sqrt(module.ER_TRANSITION_RHO_WEIGHT) * values[1],
                0.0, np.sqrt(module.ER_TRANSITION_STRENGTH_WEIGHT) * smooth_deficit,
            ])

        h = 1e-6
        fd = np.column_stack([
            (independent_residual(h * axis) - independent_residual(-h * axis)) / (2 * h)
            for axis in np.eye(2)
        ])
        np.testing.assert_allclose(actual.residuals, independent_residual(np.zeros(2)), atol=1e-12)
        np.testing.assert_allclose(actual.jacobian, fd, rtol=1e-6, atol=1e-8)
        wrapped = SimpleNamespace(
            result=actual, residuals=actual.residuals, jacobian=actual.jacobian, elapsed_s=0.0,
        )
        diagnostic = module.iteration_diagnostics(wrapped)
        assert f"Er_transition_rho={raw[0]:.8e}" in diagnostic
        assert f"Er_transition_strength={strength:.8e}" in diagnostic
        assert "Er_transition_location_cost=nan" not in diagnostic
        assert "Er_transition_strength_cost=nan" not in diagnostic
        assert "Er_left=" not in diagnostic
        assert "Er_right=" not in diagnostic


@pytest.mark.parametrize("name", [item for item in EXAMPLES if "database" in item or "finitebeta" in item])
def test_root_only_builders_forward_options_without_evolving_transport(name, monkeypatch):
    module = importlib.import_module(f"examples.optimization.{name}")
    captured = {}
    marker = object()

    def build(config, terms, **kwargs):
        captured.update(config=config, terms=terms, **kwargs)
        return marker

    monkeypatch.setattr(opt, "geometry_initial_er_root_only_least_squares_problem", build)
    args = module.parser().parse_args([])
    if "database" in name:
        problem = module.build_problem({}, Path("input.mock"), 2, args)
    else:
        problem = module.build_problem(Path("input.mock"), 2)
    assert problem is marker
    assert captured["include_profiles"] is False
    assert captured["root_options"] == module.ROOT_OPTIONS
    assert "full_transport" not in captured["reverse_stage_mode"]
    assert captured["max_mode"] == 2


@pytest.mark.parametrize("name", [
    "optimize_geometry_qi_max_er_transition_initial_root",
    "optimize_geometry_qi_max_er_transition_bootstrap_initial_root",
    "optimize_profiles_max_er_transition_bootstrap_initial_root",
])
def test_entrypoints_keep_zero_weight_transition_monitors(name, monkeypatch, tmp_path):
    module = importlib.import_module(f"examples.optimization.{name}")
    captured = {}

    class StopBeforeRuntime(Exception):
        pass

    def build(config, terms, **kwargs):
        captured.update(terms=terms, **kwargs)
        raise StopBeforeRuntime

    monkeypatch.setattr(opt, "geometry_initial_er_root_only_least_squares_problem", build)
    monkeypatch.setattr(module, "OUT_DIR", tmp_path)
    with pytest.raises(StopBeforeRuntime):
        module.main()
    names = tuple(
        term.objective.name
        for term in opt._normalize_initial_er_root_least_squares_terms(captured["terms"])
    )
    assert "Er_transition_rho" in names
    assert "Er_transition_location_moment" in names
    assert names.count("Er_transition_strength") == 2
    assert captured["root_options"] == module.ROOT_OPTIONS
    assert captured["include_profiles"] == name.startswith("optimize_profiles_")


@pytest.mark.parametrize("mode", [
    "database_root_experiment",
    "database_root_jit_experiment",
    "database_root_fresh_payload_experiment",
    "optimization_payload_root_scan_geometry_experiment",
])
def test_real_root_problem_setup_and_result_adapter_share_unique_objectives(mode, monkeypatch):
    """Exercise the public setup and result adapter that failed on the GPU.

    Only physics/setup work is stubbed. The objective ordering, result shape
    validation and expansion to separate residual rows use production code.
    """
    example = importlib.import_module(
        "examples.optimization.optimize_geometry_qi_max_er_transition_bootstrap_initial_root_database"
    )
    is_database = mode.startswith("database_")
    config = {
        "geometry": {"vmec_input_file": "input.mock", "n_radial": 5},
        "neoclassical": {"flux_model": "ntx_scan_runtime" if is_database else "ntx_exact_lij_runtime"},
        "profiles": {"n0": 4.21, "T0": 17.8, "density_shape_power": 10.0, "temperature_shape_power": 2.0},
    }
    captured = {}
    context, raw_stage = object(), object()
    runtime = SimpleNamespace()
    state = SimpleNamespace(pressure=jnp.ones((2, 5), dtype=jnp.float64))
    monkeypatch.setattr(opt, "_prepare_initial_er_root_config", lambda value, **kwargs: copy.deepcopy(value))
    monkeypatch.setattr(opt, "build_geometry_autodiff_context", lambda *args, **kwargs: context)
    monkeypatch.setattr(opt, "build_runtime_context", lambda value: (runtime, state))
    monkeypatch.setattr(opt, "freeze_physical_qi_maxj_pitches", lambda value, *args, **kwargs: value)
    monkeypatch.setattr(opt, "geometry_raw_block_stage", lambda *args, **kwargs: raw_stage)
    monkeypatch.setattr(opt, "geometry_raw_block_transpose_optimization_stage", lambda *args, **kwargs: object())
    monkeypatch.setattr(opt, "_prepare_initial_root_payload_static", lambda *args, **kwargs: object())
    monkeypatch.setattr(opt, "find_ntx_support_payload", lambda value: {"geometry": object(), "ntx_support": object()})
    monkeypatch.setattr(opt.InitialErTransportPayloadAdapter, "from_payload", lambda value: object())
    monkeypatch.setattr(opt, "build_initial_er_transport_reverse_stage", lambda **kwargs: object())

    def root_stage(**kwargs):
        captured["root_objectives"] = kwargs["objective_names"]
        return object()

    def payload_physics(**kwargs):
        captured["payload_objectives"] = kwargs["objective_labels"]
        # Synthetic raw VMEC derivative; no geometry/NTX kernel is invoked.
        return kwargs["objective_values"], kwargs["profile_gradient_matrix"], geometry_gradient

    def payload_stage(**kwargs):
        def invoke(raw, deltas, values, profiles, bars):
            kernel_result = kwargs["payload_to_vmec_impl"](raw, deltas, values, profiles, bars)
            # This genuine result constructor rejected duplicate labels in
            # the user's run. Keep it active in this regression test.
            return kwargs["result_from_kernel"](kernel_result)
        return SimpleNamespace(payload_to_vmec=invoke)

    monkeypatch.setattr(opt, "build_database_initial_root_experiment_stage", root_stage)
    monkeypatch.setattr(opt, "_optimization_payload_to_vmec_table", payload_physics)
    monkeypatch.setattr(opt, "build_initial_root_payload_assembly_stage", payload_stage)
    problem = opt.geometry_initial_er_root_only_least_squares_problem(
        config, example.TERMS, parameters="RBC:1:0", include_profiles=False,
        root_options=example.ROOT_OPTIONS, reverse_stage_mode=mode,
    )
    names = (
        "Er_transition_rho", "Er_transition_location_moment",
        "Er_transition_strength", "bootstrap_current_softmax_abs_scaled",
    )
    if is_database:
        assert captured["root_objectives"] == names
    terms = tuple(term for term in problem.terms if term.objective.family == "transport")
    assert len(terms) == 5  # Raw strength and its penalty remain distinct rows.
    profiles = jnp.zeros((4, 6), dtype=jnp.float64)
    geometry_gradient = jnp.asarray([[0.01], [0.02], [0.03], [0.04]])
    for strength in (0.2, 2.0):
        values = jnp.asarray([0.6, 0.6 - 0.514, strength, 0.3])
        assembly = problem.payload_assembly_stage.payload_to_vmec(
            object(), jnp.zeros(1), values, profiles, {},
        )
        assert captured["payload_objectives"] == names
        table = assembly.table_result
        assert table.objective_labels == names
        assert table.objective_values.shape == (4,)
        assert table.profile_gradient_matrix.shape == (4, 6)
        assert table.geometry_gradient_matrix.shape == (4, 1)
        base = reverse.assemble_least_squares_result(
            opt._base_terms_for_mixed_initial_er_root(terms),
            parameter_set=problem.parameter_set,
            backend_results={"transport": reverse.ObjectiveTableResult(
                objective_names=names, values=table.objective_values,
                jacobian=table.geometry_gradient_matrix,
            )},
        )
        evaluation = reverse.LeastSquaresEvaluation(
            result=base, residuals=base.residuals, jacobian=base.jacobian, elapsed_s=0.0,
        )
        result = opt._assemble_mixed_initial_er_root_result(terms, base_evaluation=evaluation)
        assert result.residual_labels == (
            "transport:Er_transition_rho", "transport:Er_transition_location_moment",
            "transport:Er_transition_strength", "Er_transition_strength_deficit",
            "bootstrap_current_penalty",
        )
        assert float(result.objective_values["transport:Er_transition_strength"]) == strength
        assert np.isfinite(result.jacobian).all()
        deficit = 0.9 - strength
        expected_deficit = 0.5 * (deficit + np.hypot(deficit, 1e-6))
        expected_derivative = -0.5 * (1.0 + deficit / np.hypot(deficit, 1e-6))
        np.testing.assert_allclose(result.residuals[3], 10.0 * expected_deficit, atol=1e-14)
        np.testing.assert_allclose(result.jacobian[3, 0], 10.0 * expected_derivative * 0.03, atol=1e-14)
        np.testing.assert_array_equal(result.jacobian[2], [0.0])
