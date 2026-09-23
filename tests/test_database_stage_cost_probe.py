"""Cost-probe control tests; no VMEC, database preparation or JAX compilation."""

import dataclasses
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def probe():
    path = Path(__file__).with_name("benchmark_database_reverse_stage_cost.py")
    spec = importlib.util.spec_from_file_location("database_stage_cost_probe_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_cost_probe_default_is_one_independent_kernel(probe):
    args = probe.parse_args([])
    assert args.component == "independent"
    assert not args.components and not args.support_components
    assert not args.lower_only and not args.save_results


@pytest.mark.parametrize("component", [
    "shared", "support_table", "support_flux_geometry",
    "support_equation_geometry", "support_table_and_geometry",
])
def test_stage_cost_probe_selects_one_explicit_component(probe, component):
    assert probe.parse_args(["--component", component]).component == component


@pytest.mark.parametrize("arguments", [
    ["--lower-only"], ["--save-results"], ["--repeats", "0"],
    ["--warmups", "0"], ["--n-radial", "1"], ["--dt", "nan"],
    ["--component", "shared", "--support-only"],
    ["--lower-only", "--component", "compare", "--dump-dir", "unused"],
])
def test_stage_cost_probe_rejects_ambiguous_or_invalid_runs(probe, arguments):
    with pytest.raises(SystemExit):
        probe.parse_args(arguments)


def test_stage_cost_probe_lower_only_exports_before_any_compile(probe, tmp_path):
    calls = []

    class Lowered:
        def compiler_ir(self):
            return "module { /* bounded test graph */ }"

        def compile(self):
            pytest.fail("A lower-only request must not compile the measured kernel")

    def lower(*inputs):
        calls.append(inputs)
        return Lowered()

    probe.jax = SimpleNamespace(
        block_until_ready=lambda value: value,
        jit=lambda function: SimpleNamespace(lower=lower),
    )
    args = probe.parse_args(["--lower-only", "--dump-dir", str(tmp_path)])
    assert probe.measure("support_table", lambda value: value, (3,), args) is None
    assert calls == [(3,)]
    assert "module" in (tmp_path / "support_table.stablehlo.txt").read_text()
    report = json.loads((tmp_path / "support_table.json").read_text())
    assert report["phase"] == "lowered"
    assert "compile_s" not in report and "warm_median_s" not in report


@pytest.mark.parametrize("component", ["support_flux_geometry", "support_equation_geometry"])
def test_stage_cost_probe_partial_uses_actual_scalar_stage_boundary(probe, component):
    @dataclasses.dataclass(frozen=True)
    class Physics:
        reverse_database_table_only: bool = False
        reverse_database_include_direct_geometry: bool = True
        flat_rhs_direct_database_table_pullback: object = "original table"
        flat_rhs_direct_database_table_pullback_batched: object = "old batched"
        flat_rhs_direct_database_split_support_pullback: object = "combined"

    @dataclasses.dataclass(frozen=True)
    class Carry:
        t: object
        y: object

    selected_hook, runtime = object(), SimpleNamespace(geometry="geo", database="db")
    calls = []

    def stage_driver(kernel, physics, carry, primal, bars, payload):
        assert physics.flat_rhs_direct_database_table_pullback is selected_hook
        assert physics.flat_rhs_direct_database_table_pullback_batched is None
        assert not physics.reverse_database_include_direct_geometry
        assert physics.reverse_database_table_only
        assert (carry.t, carry.y, primal.trial_dt, primal.stage_history) == (1, 2, 3, 4)
        assert bars == "residuals" and payload == {"geometry": "geo", "database": "db"}
        calls.append(component)
        return (0.0,)

    probe.solvers = SimpleNamespace(
        _radau_exact_stage_residual_database_table_support_pullback_batched=stage_driver,
        _radau_zero_support_delta_tree_like=lambda support: support,
    )
    probe.jax = SimpleNamespace(tree_util=SimpleNamespace(
        tree_flatten_with_path=lambda support: ([("geometry", 0), ("database", 0)], None),
        keystr=str,
    ))
    probe.measure = lambda name, function, inputs, args: function(*inputs)
    probe.input_fingerprint = lambda values: "fixed-seed"
    args = probe.parse_args(["--component", component])
    args.fixture_signature = {"state_inputs_sha256": "fixed-state"}
    probe.measure_support(runtime, "kernel", Physics(), Carry(0, 0),
                          (1, 2, 3, 4, 5), "residuals", args, {component: selected_hook})
    assert calls == [component]
    assert args.output_leaf_labels == ["geometry", "database"]


@pytest.mark.parametrize("change", [None, "fixture", "nonfinite", "value", "dtype"])
def test_stage_cost_probe_saved_comparison_checks_finite_values_and_fixture(probe, tmp_path, change):
    metadata = {"schema": 1, "fixture_signature": {"database_sha256": "same"}}
    expected = np.asarray([1.0, 2.0])
    actual = expected.copy()
    actual_meta = json.loads(json.dumps(metadata))
    if change == "fixture":
        actual_meta["fixture_signature"]["database_sha256"] = "different"
    elif change == "nonfinite":
        actual[0] = np.nan
    elif change == "value":
        actual[0] += 1.0
    elif change == "dtype":
        actual = actual.astype(np.float32)
    reference, candidate = tmp_path / "reference.npz", tmp_path / "candidate.npz"
    np.savez(reference, __metadata__=json.dumps(metadata), leaf_0000=expected)
    np.savez(candidate, __metadata__=json.dumps(actual_meta), leaf_0000=actual)
    if change is None:
        probe.compare_results((reference, candidate), rtol=1e-12, atol=1e-14)
    else:
        with pytest.raises(AssertionError):
            probe.compare_results((reference, candidate), rtol=1e-12, atol=1e-14)
