import argparse

import NEOPAX.cli as cli


def test_apply_cli_overrides_maps_common_runtime_flags():
    config = {
        "general": {"mode": "transport"},
        "geometry": {"vmec_file": "old_vmec.nc", "boozer_file": "old_booz.nc", "n_radial": 51},
        "energy_grid": {"n_x": 4},
        "transport_solver": {"transport_solver_backend": "theta_newton", "integrator": "theta_newton", "dt": 1.0e-4, "t_final": 1.0},
    }
    args = argparse.Namespace(
        mode="fluxes",
        vmec_file="new_vmec.nc",
        boozer_file="new_booz.nc",
        n_radial=33,
        n_x=7,
        backend="radau",
        dt=2.0e-4,
        t_final=3.5,
        output_dir="outputs/cli_case",
        set_values=[],
        config="dummy.toml",
    )

    out = cli.apply_cli_overrides(config, args)

    assert out["general"]["mode"] == "fluxes"
    assert out["geometry"]["vmec_file"] == "new_vmec.nc"
    assert out["geometry"]["boozer_file"] == "new_booz.nc"
    assert out["geometry"]["n_radial"] == 33
    assert out["energy_grid"]["n_x"] == 7
    assert out["transport_solver"]["transport_solver_backend"] == "radau"
    assert out["transport_solver"]["integrator"] == "radau"
    assert out["transport_solver"]["dt"] == 2.0e-4
    assert out["transport_solver"]["t_final"] == 3.5
    assert out["transport_output"]["transport_output_dir"] == "outputs/cli_case"
    assert out["fluxes"]["fluxes_output_dir"] == "outputs/cli_case"
    assert out["sources"]["sources_output_dir"] == "outputs/cli_case"
    assert out["ambipolarity"]["er_ambipolar_output_dir"] == "outputs/cli_case"


def test_apply_cli_overrides_supports_generic_set_values():
    config = {}
    args = argparse.Namespace(
        mode=None,
        vmec_file=None,
        boozer_file=None,
        n_radial=None,
        n_x=None,
        backend=None,
        dt=None,
        t_final=None,
        output_dir=None,
        set_values=[
            "general.mode=transport",
            "geometry.n_radial=65",
            "transport_solver.throw=true",
            "turbulence.debug_heat_flux_scale=0.25",
        ],
        config="dummy.toml",
    )

    out = cli.apply_cli_overrides(config, args)

    assert out["general"]["mode"] == "transport"
    assert out["geometry"]["n_radial"] == 65
    assert out["transport_solver"]["throw"] is True
    assert out["turbulence"]["debug_heat_flux_scale"] == 0.25


def test_cli_main_loads_config_applies_overrides_and_runs(monkeypatch, tmp_path):
    config_path = tmp_path / "case.toml"
    config_path.write_text("[general]\nmode='transport'\n", encoding="utf-8")

    loaded = {"general": {"mode": "transport"}}
    observed = {}

    def fake_load_config(path):
        observed["loaded_path"] = str(path)
        return loaded

    def fake_run_config(config):
        observed["mode"] = config["general"]["mode"]
        observed["n_radial"] = config["geometry"]["n_radial"]
        return {"ok": True}

    monkeypatch.setattr(cli, "load_config", fake_load_config)
    monkeypatch.setattr(cli, "run_config", fake_run_config)

    rc = cli.main([str(config_path), "--mode", "fluxes", "--n-radial", "41"])

    assert rc == 0
    assert observed["loaded_path"] == str(config_path.resolve())
    assert observed["mode"] == "fluxes"
    assert observed["n_radial"] == 41
