"""Array recording only; no VMEX, NTX, roots or transport solves."""
from unittest.mock import patch
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from NEOPAX._optimization_diagnostics import save_gradient_audit


def test_disabled_audit_does_not_inspect_or_copy_inputs(monkeypatch):
    monkeypatch.delenv("NEOPAX_GRADIENT_AUDIT_DIR", raising=False)
    class Unreadable:
        def __array__(self, *args, **kwargs):
            raise AssertionError("Disabled audit inspected data")
    with patch("jax.device_get", side_effect=AssertionError("Disabled audit transferred data")):
        assert save_gradient_audit("disabled", value=Unreadable()) is None


def test_audit_preserves_named_values_and_repeated_calls(tmp_path, monkeypatch):
    monkeypatch.setenv("NEOPAX_GRADIENT_AUDIT_DIR", str(tmp_path / "nested"))
    inputs = {"state": {"y": np.asarray([1., np.nan, np.inf])},
              "labels": ("Er", "bootstrap"), "mask": np.asarray([True, False])}
    first = save_gradient_audit("terminal/audit", **inputs)
    second = save_gradient_audit("terminal/audit", **inputs)
    assert first != second and first.parent == tmp_path / "nested"
    with np.load(first, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["state['y']"], inputs["state"]["y"])
        np.testing.assert_array_equal(data["mask"], inputs["mask"])
        assert data["labels[0]"].item() == "Er"
        assert data["labels[1]"].item() == "bootstrap"


def test_audit_skips_integer_cotangent_placeholders(tmp_path, monkeypatch):
    import jax
    monkeypatch.setenv("NEOPAX_GRADIENT_AUDIT_DIR", str(tmp_path))
    path = save_gradient_audit("bars", zero=np.zeros(2, dtype=jax.dtypes.float0), real=np.ones(2))
    with np.load(path, allow_pickle=False) as data:
        assert data.files == ["real"]


def test_audit_rejects_unrecoverable_object_leaves(tmp_path, monkeypatch):
    monkeypatch.setenv("NEOPAX_GRADIENT_AUDIT_DIR", str(tmp_path))
    with pytest.raises(TypeError, match="Unsupported gradient audit leaf"):
        save_gradient_audit("bad", value=object())
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("profile", [True, False])
def test_saved_chain_comparison_checks_the_point_and_terminal_chain_rule(profile):
    path = Path(__file__).resolve().parents[1] / "examples/benchmarks/compare_transport_gradient_audit.py"
    spec = importlib.util.spec_from_file_location("audit_comparison", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    class Archive(dict):
        @property
        def files(self):
            return list(self)
    baseline = Archive(initial_y=np.zeros(2), final_y=np.asarray([.4, -.2]),
                       final_time=np.asarray(.1), values=np.asarray([-.44]),
                       **{"labels[0]": np.asarray("objective")})
    ad = Archive(baseline, final_y_bars=np.asarray([[.8, 3.]]))
    h, direction = 1e-4, np.asarray([2., -1.])
    minus = Archive(baseline, final_y=baseline["final_y"]-h*direction)
    plus = Archive(baseline, final_y=baseline["final_y"]+h*direction)
    report = dict(objective_labels=["objective"], objective_values=[-.44], fd_step=h, gradient_fd=[-1.4],
                  baseline_geometry_final_state_fd=[-1.4],
                  parameter_kind="profile" if profile else "realtime_geometry")
    row, = module.compare(ad, baseline, minus, plus, report)
    assert row["terminal_bar_dot_fd_final_y"] == pytest.approx(-1.4, abs=1e-12)
    assert row["relative_difference"] < 1e-12
    # Equal printed objective values must not clear a different plasma state.
    different = Archive(baseline, final_y=baseline["final_y"]+.001)
    with pytest.raises(ValueError, match="baseline final_y differs"):
        module.compare(ad, different, minus, plus, report)
    with pytest.raises(ValueError, match="labels/order differ"):
        module.compare(ad, baseline, minus, plus, dict(report, objective_labels=["wrong"]))
    with pytest.raises(ValueError, match="summary does not match"):
        module.compare(ad, baseline, minus, plus, dict(report, objective_values=[1.]))
