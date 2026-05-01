import contextlib
import io
from pathlib import Path

import h5py
import jax.numpy as jnp
import pytest

from NEOPAX._entropy_models import get_entropy_model
from NEOPAX._transport_flux_models import (
    AnalyticalTurbulentTransportModel,
    CombinedTransportFluxModel,
    FluxesRFileTransportModel,
    PowerAnalyticalTurbulentTransportModel,
    build_fluxes_r_file_transport_model,
    read_flux_profile_file,
)
from NEOPAX._fem import cell_centered_from_faces, faces_from_cell_centered
from NEOPAX._orchestrator import calculate_fluxes_from_config
from NEOPAX._state import TransportState


class DummySpecies:
    number_species = 2


class DummyGeometry:
    def __init__(self):
        self.r_grid_half = jnp.array([0.0, 0.5, 1.0])
        self.r_grid = jnp.array([0.25, 0.75])


class DummyFluxModel:
    def __init__(self, gamma, q, upar):
        self.gamma = jnp.asarray(gamma)
        self.q = jnp.asarray(q)
        self.upar = jnp.asarray(upar)

    def __call__(self, state):
        del state
        return {"Gamma": self.gamma, "Q": self.q, "Upar": self.upar}

    def build_local_particle_flux_evaluator(self, state):
        del state

        def evaluator(radius_index, er_value):
            del er_value
            return self.gamma[:, radius_index]

        return evaluator

    def evaluate_face_fluxes(self, state, face_state, **kwargs):
        del state, face_state, kwargs
        return {"Gamma": self.gamma, "Q": self.q, "Upar": self.upar}


def test_transport_flux_base_lagged_response_is_flux_linearization():
    from NEOPAX._transport_flux_models import TransportFluxModelBase

    class LinearFluxModel(TransportFluxModelBase):
        def __call__(self, state, geometry=None, params=None):
            del geometry, params
            gamma = 2.0 * state.density
            q = 3.0 * state.pressure
            upar = 4.0 * state.Er[None, :]
            return {"Gamma": gamma, "Q": q, "Upar": upar}

    model = LinearFluxModel()
    state0 = TransportState(
        density=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        pressure=jnp.array([[5.0, 6.0], [7.0, 8.0]]),
        Er=jnp.array([0.25, 0.5]),
    )
    state1 = TransportState(
        density=state0.density + 0.1,
        pressure=state0.pressure - 0.2,
        Er=state0.Er + 0.05,
    )

    lagged = model.build_lagged_response(state0)
    out = model.evaluate_with_lagged_response(state1, lagged)
    exact = model(state1)

    assert jnp.allclose(out["Gamma"], exact["Gamma"])
    assert jnp.allclose(out["Q"], exact["Q"])
    assert jnp.allclose(out["Upar"], exact["Upar"])


def _write_flux_file(path: Path, r, gamma=None, q=None, upar=None):
    with h5py.File(path, "w") as f:
        f["r"] = jnp.asarray(r)
        if gamma is not None:
            f["Gamma"] = jnp.asarray(gamma)
        if q is not None:
            f["Q"] = jnp.asarray(q)
        if upar is not None:
            f["Upar"] = jnp.asarray(upar)


def test_read_flux_profile_file_accepts_1d_and_2d_inputs(tmp_path):
    path = tmp_path / "fluxes.h5"
    _write_flux_file(
        path,
        r=[0.0, 0.5, 1.0],
        gamma=[1.0, 2.0, 3.0],
        q=[[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
    )

    r_data, gamma_data, q_data, upar_data = read_flux_profile_file(path, n_species=2)
    assert tuple(r_data.shape) == (3,)
    assert tuple(gamma_data.shape) == (2, 3)
    assert tuple(q_data.shape) == (2, 3)
    assert upar_data is None
    assert jnp.allclose(gamma_data[0], jnp.array([1.0, 2.0, 3.0]))
    assert jnp.allclose(gamma_data[1], jnp.array([1.0, 2.0, 3.0]))


def test_fluxes_r_file_model_cell_centered_reconstructs_faces(tmp_path):
    path = tmp_path / "cell_fluxes.h5"
    gamma = jnp.array([[1.0, 3.0], [2.0, 4.0]])
    q = jnp.array([[10.0, 30.0], [20.0, 40.0]])
    upar = jnp.array([[0.5, 1.5], [1.0, 2.0]])
    _write_flux_file(path, r=[0.25, 0.75], gamma=gamma, q=q, upar=upar)

    with contextlib.redirect_stdout(io.StringIO()):
        model = build_fluxes_r_file_transport_model(
            DummySpecies(),
            DummyGeometry(),
            fluxes_file=path,
            grid_location="cell_centered",
        )

    center_fluxes = model(state=None)
    face_fluxes = model.evaluate_face_fluxes(state=None, face_state=None)

    assert jnp.allclose(center_fluxes["Gamma"], gamma)
    assert jnp.allclose(center_fluxes["Q"], q)
    assert jnp.allclose(face_fluxes["Gamma"], jnp.vstack([faces_from_cell_centered(gamma[0]), faces_from_cell_centered(gamma[1])]))
    assert jnp.allclose(face_fluxes["Q"], jnp.vstack([faces_from_cell_centered(q[0]), faces_from_cell_centered(q[1])]))


def test_fluxes_r_file_heat_flux_scaling_applies_only_to_q(tmp_path):
    path = tmp_path / "scaled_fluxes.h5"
    gamma = jnp.array([[1.0, 3.0], [2.0, 4.0]])
    q = jnp.array([[10.0, 30.0], [20.0, 40.0]])
    upar = jnp.array([[0.5, 1.5], [1.0, 2.0]])
    _write_flux_file(path, r=[0.25, 0.75], gamma=gamma, q=q, upar=upar)

    with contextlib.redirect_stdout(io.StringIO()):
        model = build_fluxes_r_file_transport_model(
            DummySpecies(),
            DummyGeometry(),
            fluxes_file=path,
            grid_location="cell_centered",
            debug_heat_flux_scale=2.5,
        )

    center_fluxes = model(state=None)
    face_fluxes = model.evaluate_face_fluxes(state=None, face_state=None)

    assert jnp.allclose(center_fluxes["Gamma"], gamma)
    assert jnp.allclose(center_fluxes["Upar"], upar)
    assert jnp.allclose(center_fluxes["Q"], 2.5 * q)
    assert jnp.allclose(face_fluxes["Gamma"], jnp.vstack([faces_from_cell_centered(gamma[0]), faces_from_cell_centered(gamma[1])]))
    assert jnp.allclose(face_fluxes["Q"], 2.5 * jnp.vstack([faces_from_cell_centered(q[0]), faces_from_cell_centered(q[1])]))


def test_fluxes_r_file_model_face_centered_reconstructs_cells(tmp_path):
    path = tmp_path / "face_fluxes.h5"
    gamma_faces = jnp.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
    q_faces = jnp.array([[10.0, 30.0, 50.0], [20.0, 40.0, 60.0]])
    _write_flux_file(path, r=[0.0, 0.5, 1.0], gamma=gamma_faces, q=q_faces)

    with contextlib.redirect_stdout(io.StringIO()):
        model = FluxesRFileTransportModel(
            species=DummySpecies(),
            geometry=DummyGeometry(),
            r_data=jnp.array([0.0, 0.5, 1.0]),
            gamma_data=gamma_faces,
            q_data=q_faces,
            upar_data=None,
            profile_location="face_centered",
        )

    center_fluxes = model(state=None)
    assert jnp.allclose(center_fluxes["Gamma"], jnp.vstack([cell_centered_from_faces(gamma_faces[0]), cell_centered_from_faces(gamma_faces[1])]))
    assert jnp.allclose(center_fluxes["Q"], jnp.vstack([cell_centered_from_faces(q_faces[0]), cell_centered_from_faces(q_faces[1])]))


def test_fluxes_r_file_with_q_scale_returns_updated_model(tmp_path):
    path = tmp_path / "scale_update_fluxes.h5"
    gamma = jnp.array([[1.0, 3.0], [2.0, 4.0]])
    q = jnp.array([[10.0, 30.0], [20.0, 40.0]])
    _write_flux_file(path, r=[0.25, 0.75], gamma=gamma, q=q)

    with contextlib.redirect_stdout(io.StringIO()):
        model = build_fluxes_r_file_transport_model(
            DummySpecies(),
            DummyGeometry(),
            fluxes_file=path,
            grid_location="cell_centered",
        )

    updated = model.with_q_scale(0.5)

    assert model.q_scale == 1.0
    assert updated.q_scale == 0.5
    assert jnp.allclose(updated(state=None)["Q"], 0.5 * q)


def test_fluxes_r_file_invalid_profile_location_raises():
    model = FluxesRFileTransportModel(
        species=DummySpecies(),
        geometry=DummyGeometry(),
        r_data=jnp.array([0.0, 1.0]),
        gamma_data=jnp.ones((2, 2)),
        q_data=None,
        upar_data=None,
        profile_location="diagonal",
    )
    with pytest.raises(ValueError):
        model._normalize_profile_location()


def test_analytical_turbulent_transport_model_with_transport_coeffs_updates_coefficients():
    model = AnalyticalTurbulentTransportModel(
        species="species",
        grid="grid",
        chi_t=jnp.array([1.0, 2.0]),
        chi_n=jnp.array([3.0, 4.0]),
        field="field",
    )

    updated = model.with_transport_coeffs(chi_t=jnp.array([5.0, 6.0]))

    assert jnp.allclose(model.chi_t, jnp.array([1.0, 2.0]))
    assert jnp.allclose(updated.chi_t, jnp.array([5.0, 6.0]))
    assert jnp.allclose(updated.chi_n, jnp.array([3.0, 4.0]))


def test_power_analytical_turbulent_transport_model_with_transport_coeffs_updates_inputs():
    model = PowerAnalyticalTurbulentTransportModel(
        species="species",
        field="field",
        chi_t=jnp.array([1.0, 2.0]),
        chi_n=jnp.array([3.0, 4.0]),
        pressure_source_model="source_a",
        total_power_mw=5.0,
    )

    updated = model.with_transport_coeffs(
        chi_n=jnp.array([7.0, 8.0]),
        pressure_source_model="source_b",
        total_power_mw=9.0,
    )

    assert jnp.allclose(model.chi_n, jnp.array([3.0, 4.0]))
    assert jnp.allclose(updated.chi_t, jnp.array([1.0, 2.0]))
    assert jnp.allclose(updated.chi_n, jnp.array([7.0, 8.0]))
    assert updated.pressure_source_model == "source_b"
    assert updated.total_power_mw == 9.0


def test_combined_transport_flux_model_can_drop_turbulent_particle_flux():
    gamma_neo = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    gamma_turb = jnp.array([[2.0, 2.0], [2.0, 2.0]])
    gamma_classical = jnp.array([[3.0, 3.0], [3.0, 3.0]])
    q_neo = jnp.array([[10.0, 10.0], [10.0, 10.0]])
    q_turb = jnp.array([[20.0, 20.0], [20.0, 20.0]])
    q_classical = jnp.array([[30.0, 30.0], [30.0, 30.0]])

    model = CombinedTransportFluxModel(
        neoclassical_model=DummyFluxModel(gamma_neo, q_neo, jnp.zeros_like(gamma_neo)),
        turbulent_model=DummyFluxModel(gamma_turb, q_turb, jnp.zeros_like(gamma_turb)),
        classical_model=DummyFluxModel(gamma_classical, q_classical, jnp.zeros_like(gamma_classical)),
        include_turbulent_particle_flux=False,
    )

    out = model(state=None)
    assert jnp.allclose(out["Gamma"], gamma_neo + gamma_classical)
    assert jnp.allclose(out["Gamma_turb"], jnp.zeros_like(gamma_turb))
    assert jnp.allclose(out["Q"], q_neo + q_turb + q_classical)

    face_fluxes = model.evaluate_face_fluxes(state=None, face_state=None)
    assert jnp.allclose(face_fluxes["Gamma_turb"], jnp.zeros_like(gamma_turb))
    assert jnp.allclose(face_fluxes["Gamma"], gamma_neo + gamma_classical)

    local_eval = model.build_local_particle_flux_evaluator(state=None)
    gamma_local = local_eval(0, 0.0)
    assert jnp.allclose(gamma_local, gamma_neo[:, 0] + gamma_classical[:, 0])


def test_calculate_fluxes_from_config_uses_flux_output_flags():
    flux_model = lambda state: {"Gamma": jnp.asarray([[1.0]]), "Q": jnp.asarray([[2.0]]), "Upar": jnp.asarray([[3.0]])}
    config = {
        "fluxes": {
            "fluxes_plot": True,
            "fluxes_write_hdf5": True,
            "fluxes_output_dir": "./outputs/unit_flux",
            "fluxes_reference_file": "./ref.h5",
            "fluxes_reference_label": "NTSS",
        }
    }
    params = {"species": None, "energy_grid": None, "geometry": None, "database": None}

    fluxes, do_plot, do_hdf5, output_dir, overlay_reference, reference_file, reference_label = calculate_fluxes_from_config(
        state=None,
        config=config,
        params=params,
        flux_model=flux_model,
    )

    assert do_plot is True
    assert do_hdf5 is True
    assert output_dir == "./outputs/unit_flux"
    assert overlay_reference is True
    assert reference_file == "./ref.h5"
    assert reference_label == "NTSS"
    assert jnp.allclose(fluxes["Gamma"], jnp.asarray([[1.0]]))


def test_fluxes_r_file_entropy_alias_is_registered():
    assert get_entropy_model("fluxes_r_file") is get_entropy_model("ntx_database")
