from pathlib import Path

import NEOPAX.api as api


def test_prepare_config_accepts_path_and_common_overrides(tmp_path, monkeypatch):
    config_path = tmp_path / "case.toml"
    config_path.write_text("[general]\nmode='transport'\n", encoding="utf-8")

    monkeypatch.setattr(api, "load_config", lambda path: {"general": {"mode": "transport"}})

    out = api.prepare_config(
        config_path,
        mode="fluxes",
        n_radial=33,
        n_x=6,
        backend="radau",
        dt=2.0e-4,
        t_final=4.0,
        output_dir="outputs/api",
        set_values=["turbulence.debug_heat_flux_scale=0.5"],
    )

    assert out["general"]["mode"] == "fluxes"
    assert out["geometry"]["n_radial"] == 33
    assert out["energy_grid"]["n_x"] == 6
    assert out["transport_solver"]["transport_solver_backend"] == "radau"
    assert out["transport_solver"]["dt"] == 2.0e-4
    assert out["transport_solver"]["t_final"] == 4.0
    assert out["transport_output"]["transport_output_dir"] == "outputs/api"
    assert out["turbulence"]["debug_heat_flux_scale"] == 0.5


def test_run_wraps_raw_result_and_extracts_final_state(monkeypatch):
    monkeypatch.setattr(api, "prepare_config", lambda *args, **kwargs: {"general": {"mode": "transport"}})
    monkeypatch.setattr(
        api,
        "run_config",
        lambda config: {
            "final_state": {"density": "dummy"},
            "ys": ["state0", "state1"],
            "ts": [0.0, 1.0],
            "dts": [0.1, 0.2],
            "accepted_mask": [True, True],
            "failed_mask": [False, False],
            "fail_codes": [0, 0],
            "n_steps": 2,
            "done": True,
            "failed": False,
            "fail_code": 0,
            "final_time": 1.0,
            "rho": [0.0, 0.5, 1.0],
            "output_dir": "outputs/run",
        },
    )

    result = api.run("dummy.toml", mode="transport")

    assert result.mode == "transport"
    assert result.final_state == {"density": "dummy"}
    assert result.saved_states == ["state0", "state1"]
    assert result.time_grid == [0.0, 1.0]
    assert result.saved_step_sizes == [0.1, 0.2]
    assert result.accepted_mask == [True, True]
    assert result.failed_mask == [False, False]
    assert result.fail_codes == [0, 0]
    assert result.n_steps == 2
    assert result.done is True
    assert result.failed is False
    assert result.fail_code == 0
    assert result.final_time == 1.0
    assert result.rho == [0.0, 0.5, 1.0]
    assert result.output_dir == Path("outputs/run")
    assert isinstance(result.raw_result, dict)
