from pathlib import Path

import h5py
import jax.numpy as jnp

from NEOPAX.main import (
    _load_ntss_reference_profiles,
    _normalize_solver_config,
    _resolve_reference_path,
)


def test_normalize_solver_config_prefers_transport_solver_section():
    config = {
        "transport_solver": {
            "transport_solver_backend": "theta_newton",
            "density_floor": 2.5e-6,
        },
        "solver": {
            "integrator": "radau",
        },
        "neoclassical": {"flux_model": "monkes_database"},
        "turbulence": {"flux_model": "none"},
    }

    out = _normalize_solver_config(config)
    assert out["transport_solver_backend"] == "theta_newton"
    assert out["integrator"] == "theta_newton"
    assert out["neoclassical_flux_model"] == "monkes_database"
    assert out["turbulence_flux_model"] == "none"
    assert out["density_floor"] == 2.5e-6
    assert out["Er_relax"] == 1.0
    assert out["DEr"] == 1.0


def test_normalize_solver_config_falls_back_to_legacy_solver_section():
    config = {
        "solver": {
            "integrator": "radau",
        },
        "neoclassical": {"flux_model": "none"},
        "turbulence": {"flux_model": "turbulent_power_analytical"},
    }

    out = _normalize_solver_config(config)
    assert out["transport_solver_backend"] == "radau"
    assert out["integrator"] == "radau"
    assert out["density_floor"] == 1.0e-6
    assert out["turbulence_flux_model"] == "turbulent_power_analytical"


def test_resolve_reference_path_handles_relative_paths(tmp_path, monkeypatch):
    ref = tmp_path / "ref.h5"
    ref.write_bytes(b"test")
    monkeypatch.chdir(tmp_path)

    resolved = _resolve_reference_path("ref.h5")
    assert resolved == ref.resolve()


def test_resolve_reference_path_returns_none_for_missing_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _resolve_reference_path("missing.h5") is None


def test_load_ntss_reference_profiles_interpolates_scalar_and_species_profiles(tmp_path, monkeypatch):
    path = tmp_path / "profiles.h5"
    with h5py.File(path, "w") as f:
        f["r"] = jnp.array([0.0, 0.5, 1.0])
        f["Er"] = jnp.array([0.0, 1.0, 2.0])
        f["ne"] = jnp.array([10.0, 20.0, 30.0])
        f["nD"] = jnp.array([1.0, 2.0, 3.0])
        f["Te"] = jnp.array([100.0, 200.0, 300.0])
        f["TD"] = jnp.array([400.0, 500.0, 600.0])
        f["Tt"] = jnp.array([700.0, 800.0, 900.0])
        f["Vr"] = jnp.ones(3)
        f["FluxQe"] = jnp.array([7.0, 8.0, 9.0])
        f["FluxQI"] = jnp.array([4.0, 5.0, 6.0])

    monkeypatch.chdir(tmp_path)
    rho = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
    out = _load_ntss_reference_profiles("profiles.h5", rho)

    assert jnp.allclose(out["Er"], jnp.array([0.0, 0.5, 1.0, 1.5, 2.0]))
    assert jnp.allclose(out["density"]["e"], jnp.array([10.0, 15.0, 20.0, 25.0, 30.0]))
    assert jnp.allclose(out["density"]["D"], jnp.array([1.0, 1.5, 2.0, 2.5, 3.0]))
    assert jnp.allclose(out["density"]["T"], jnp.array([1.0, 1.5, 2.0, 2.5, 3.0]))
    assert jnp.allclose(out["temperature"]["e"], jnp.array([100.0, 150.0, 200.0, 250.0, 300.0]))
    assert jnp.allclose(out["temperature"]["D"], jnp.array([400.0, 450.0, 500.0, 550.0, 600.0]))
    assert jnp.allclose(out["temperature"]["T"], jnp.array([700.0, 750.0, 800.0, 850.0, 900.0]))
    assert jnp.allclose(out["flux_species"]["Q_total"]["e"], jnp.array([7.0, 7.5, 8.0, 8.5, 9.0]))
