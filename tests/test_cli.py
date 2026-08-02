import argparse

import jax.numpy as jnp
import pytest

import NEOPAX.cli as cli
from NEOPAX._transport_solvers import build_time_solver


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


@pytest.mark.parametrize(
    ("result", "expected_rc"),
    [
        # Failure flag alone, with the other two fields healthy.
        ({"failed": True, "n_steps": 12, "done": True, "fail_code": 3}, 1),
        # Zero accepted steps short of t_final, the silent case reported in #5.
        ({"failed": False, "n_steps": 0, "done": False}, 1),
        # Stopped before t_final after making progress.
        ({"failed": False, "n_steps": 12, "done": False}, 1),
        # Completed solve.
        ({"failed": False, "n_steps": 12, "done": True}, 0),
        # Zero-duration solve, where t0 == t_final leaves nothing to step.
        ({"failed": False, "n_steps": 0, "done": True}, 0),
        # Non-transport mode, which carries no solver verdict to act on.
        ({"rho": [0.0, 1.0], "fluxes": {}}, 0),
    ],
)
def test_cli_main_exit_code_follows_the_solver_verdict(monkeypatch, tmp_path, result, expected_rc):
    config_path = tmp_path / "case.toml"
    config_path.write_text("[general]\nmode='transport'\n", encoding="utf-8")

    monkeypatch.setattr(cli, "load_config", lambda path: {"general": {"mode": "transport"}})
    monkeypatch.setattr(cli, "run_config", lambda config: result)

    assert cli.main([str(config_path)]) == expected_rc


# max_steps of 4 cannot cover t_final, so diffrax returns an unsuccessful solution rather than raising.
@pytest.mark.parametrize(("max_steps", "reason_expected"), [(4, True), (4096, False)])
def test_failed_solve_reason_reads_the_diffrax_solution_result(max_steps, reason_expected):
    pytest.importorskip("diffrax")

    solver = build_time_solver(
        {
            "t0": 0.0,
            "t_final": 1.0,
            "dt": 1.0e-3,
            "transport_solver_backend": "diffrax_kvaerno5",
            "max_steps": max_steps,
            "save_n": 2,
            "rtol": 1.0e-6,
            "atol": 1.0e-8,
        }
    )
    solution = solver.solve(jnp.array([1.0]), lambda t, y: -2.0 * y)

    assert not isinstance(solution, dict)
    assert (cli._failed_solve_reason(solution) is not None) is reason_expected
