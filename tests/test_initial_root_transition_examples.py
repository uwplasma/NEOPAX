"""Root-only example wiring checks without VMEC, NTX, roots, or transport solves."""

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
