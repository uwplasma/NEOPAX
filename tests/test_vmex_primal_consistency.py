import numpy as np
import pytest

from examples.benchmarks.check_vmex_primal_consistency import compare_states


def test_state_comparison_detects_a_changed_equilibrium():
    a = {"R_cos": np.asarray([[3., 4.]]), "Z_sin": np.zeros((1, 2))}
    b = {"R_cos": np.asarray([[3., 5.]]), "Z_sin": np.zeros((1, 2))}
    rows = compare_states(a, b)
    assert rows[0]["max_abs_difference"] == 1.
    assert rows[0]["relative_l2_difference"] == .2
    assert rows[1]["max_abs_difference"] == rows[1]["relative_l2_difference"] == 0.
    assert all(row["max_abs_difference"] == 0. for row in compare_states(a, a))


def test_state_comparison_rejects_invalid_reference_arrays():
    a = {"R_cos": np.ones((1, 2))}
    for b, error in (({}, ValueError), ({"R_cos": np.ones((2, 2))}, ValueError),
                     ({"R_cos": np.asarray([[1., np.nan]])}, FloatingPointError)):
        with pytest.raises(error):
            compare_states(a, b)


def test_diagnostic_writes_solver_state_and_boozer_comparisons(monkeypatch, tmp_path):
    import contextlib
    import dataclasses
    import sys
    import types
    from examples.benchmarks import check_vmex_primal_consistency as diagnostic

    @dataclasses.dataclass
    class State:
        R_cos: np.ndarray

    calls = []
    forward, implicit = State(np.asarray([[3., 4.]])), State(np.asarray([[3., 5.]]))
    backend = types.ModuleType("NEOPAX._geometry_autodiff")
    backend.build_geometry_autodiff_context = lambda path, **kw: (path, kw)

    def solve_forward(context, delta, **kwargs):
        calls.append((context, delta, kwargs))
        return forward

    def solve_implicit(context, delta, specs, **kwargs):
        calls.append((context, delta, specs, kwargs))
        return types.SimpleNamespace(state=implicit)

    backend._solve_state_for_single_param = solve_forward
    backend.geometry_raw_block_solve_from_param_vector = solve_implicit
    backend._vmec_booz_scalar_observables_from_state = lambda ctx, state: {"qi": state.R_cos.sum()}
    backend._vmec_scalar_observables_from_state = lambda *args: pytest.fail("Boozer option was ignored")
    fake_jax = types.ModuleType("jax")
    fake_jax.config = types.SimpleNamespace(update=lambda *args: None)
    fake_jax.devices = lambda platform: ["gpu"] if platform == "gpu" else []
    fake_jax.default_device = lambda device: contextlib.nullcontext()
    fake_jax.block_until_ready = lambda value: value
    fake_jax.__version__ = "test"
    fake_jax.numpy = np
    fake_vmex = types.ModuleType("vmex")
    fake_vmex.__file__ = "test/vmex.py"
    for name, module in (("jax", fake_jax), ("jax.numpy", np), ("vmex", fake_vmex),
                         ("NEOPAX._geometry_autodiff", backend)):
        monkeypatch.setitem(sys.modules, name, module)

    config = tmp_path / "input.toml"
    config.write_text('[geometry]\nvmec_input_file="input.vmec"\nvmec_lane="ad"\n'
                      'vmec_param_family="ZBS"\nvmec_param_m=2\nvmec_param_n=1\nvmec_param_delta=0.02\n')
    output = tmp_path / "result"
    monkeypatch.setattr(sys, "argv", ["check", "--config", str(config), "--out-dir", str(output),
                                      "--include-boozer"])
    diagnostic.main()
    assert calls[0][0] == calls[1][0]
    assert calls[0][1] == calls[1][1][0] == .02
    assert calls[0][2]["lane"] == "forward"
    assert calls[1][2] == (("ZBS", 2, 1),)
    with np.load(output / "states.npz") as saved:
        np.testing.assert_array_equal(saved["forward__R_cos"], forward.R_cos)
        np.testing.assert_array_equal(saved["implicit__R_cos"], implicit.R_cos)
    assert "qi,7.0,8.0,1.0" in (output / "objective_comparison.csv").read_text()
    assert "R_cos,1.0,0.2" in (output / "state_comparison.csv").read_text()
