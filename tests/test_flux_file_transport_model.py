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


class DummyFDGeometry:
    def __init__(self):
        self.r_grid_half = jnp.array([0.0, 0.5, 1.0])
        self.r_grid = jnp.array([0.25, 0.75])
        self.dr = 0.5
        self.a_b = 1.0


def _fd_model(**overrides):
    """Two-species, two-radius FD model with one temperature perturbation on species 0."""
    kwargs = dict(
        species=DummySpecies(),
        geometry=DummyFDGeometry(),
        r_data=jnp.array([0.25, 0.75]),
        gamma_data=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        q_data=jnp.array([[10.0, 20.0], [30.0, 40.0]]),
        upar_data=jnp.zeros((2, 2)),
        profile_location="cell_centered",
        lagged_response_mode="fd",
        gamma_perturb_data=jnp.array([[[1.5, 2.5], [3.0, 4.0]]]),
        q_perturb_data=jnp.array([[[14.0, 26.0], [30.0, 40.0]]]),
        perturb_delta_data=jnp.array([[0.5, 0.5]]),
        perturb_present_data=jnp.array([[True, True]]),
        perturb_kind_codes=jnp.array([1]),
        perturb_species_indices=jnp.array([0]),
    )
    kwargs.update(overrides)
    return FluxesRFileTransportModel(**kwargs)


def _fd_states():
    """Peaked reference state plus two successively flatter ones, all fluxes staying positive."""
    density = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    state0 = TransportState(
        density=density,
        pressure=jnp.array([[4.0, 2.0], [4.0, 2.0]]),
        Er=jnp.array([0.0, 0.0]),
    )
    state1 = TransportState(
        density=density,
        pressure=jnp.array([[3.8, 2.1], [4.0, 2.0]]),
        Er=jnp.array([0.0, 0.0]),
    )
    state2 = TransportState(
        density=density,
        pressure=jnp.array([[3.6, 2.2], [4.0, 2.0]]),
        Er=jnp.array([0.0, 0.0]),
    )
    return state0, state1, state2


def test_fd_lagged_response_rebuild_keeps_the_original_anchor():
    model = _fd_model()
    state0, state1, _ = _fd_states()

    response0 = model.build_lagged_response(state0)
    expected = model.evaluate_with_lagged_response(state1, response0)

    response1 = model.build_lagged_response(state1, previous_response=response0)
    rebuilt = model.evaluate_with_lagged_response(state1, response1)

    assert jnp.allclose(rebuilt["Q"], expected["Q"])
    assert jnp.allclose(rebuilt["Gamma"], expected["Gamma"])


def test_fd_lagged_response_rebuild_does_not_snap_back_to_the_file_flux():
    model = _fd_model()
    state0, state1, _ = _fd_states()

    response0 = model.build_lagged_response(state0)
    response1 = model.build_lagged_response(state1, previous_response=response0)
    rebuilt = model.evaluate_with_lagged_response(state1, response1)

    file_flux = model(state=None)
    assert bool(jnp.all(rebuilt["Q"][0] < file_flux["Q"][0]))


def test_fd_lagged_response_is_the_same_function_before_and_after_a_rebuild():
    model = _fd_model()
    state0, state1, state2 = _fd_states()

    response0 = model.build_lagged_response(state0)
    response1 = model.build_lagged_response(state1, previous_response=response0)

    through_original = model.evaluate_with_lagged_response(state2, response0)
    through_rebuilt = model.evaluate_with_lagged_response(state2, response1)

    assert jnp.allclose(through_rebuilt["Q"], through_original["Q"], rtol=0.0, atol=0.0)
    assert jnp.allclose(through_rebuilt["Gamma"], through_original["Gamma"], rtol=0.0, atol=0.0)


def test_fd_lagged_response_without_a_previous_response_anchors_at_the_given_state():
    model = _fd_model()
    state0, state1, _ = _fd_states()

    response = model.build_lagged_response(state1)
    at_anchor = model.evaluate_with_lagged_response(state1, response)

    file_flux = model(state=None)
    assert jnp.allclose(at_anchor["Q"], file_flux["Q"])
    assert not jnp.allclose(
        model.build_lagged_response(state0).reference_basis,
        response.reference_basis,
    )


def test_combined_flux_model_routes_the_previous_response_to_each_submodel():
    from NEOPAX._transport_flux_models import TransportFluxModelBase

    class ZeroFluxModel(TransportFluxModelBase):
        def __call__(self, state, geometry=None, params=None) -> dict:
            del geometry, params
            zeros = jnp.zeros_like(state.density)
            return {"Gamma": zeros, "Q": zeros, "Upar": zeros}

    fd_model = _fd_model()
    combined = CombinedTransportFluxModel(
        neoclassical_model=ZeroFluxModel(),
        turbulent_model=fd_model,
        classical_model=ZeroFluxModel(),
    )
    state0, state1, _ = _fd_states()

    response0 = combined.build_lagged_response(state0)
    response1 = combined.build_lagged_response(state1, previous_response=response0)

    expected = fd_model.evaluate_with_lagged_response(state1, response0.turbulent_response)
    rebuilt = fd_model.evaluate_with_lagged_response(state1, response1.turbulent_response)

    assert jnp.allclose(rebuilt["Q"], expected["Q"])


def test_equation_system_forwards_the_previous_response_to_the_flux_model():
    from NEOPAX._transport_equations import ComposedEquationSystem

    fd_model = _fd_model()
    system = ComposedEquationSystem(equations=(), shared_flux_model=fd_model)
    state0, state1, _ = _fd_states()

    response0 = system.build_lagged_response(state0)
    response1 = system.build_lagged_response(state1, previous_response=response0)

    expected = fd_model.evaluate_with_lagged_response(state1, response0.flux_response)
    rebuilt = fd_model.evaluate_with_lagged_response(state1, response1.flux_response)

    assert not jnp.allclose(expected["Q"], fd_model(state=None)["Q"])
    assert jnp.allclose(rebuilt["Q"], expected["Q"])


def test_non_fd_lagged_response_still_rebuilds_at_the_current_state():
    model = _fd_model(lagged_response_mode="none")
    state0, state1, _ = _fd_states()

    response0 = model.build_lagged_response(state0)
    response1 = model.build_lagged_response(state1, previous_response=response0)

    assert jnp.allclose(response1.reference_state.pressure, state1.pressure)


def _response_tangent(response, value):
    """Tangent tree matching a lagged response, with float0 on its integer and boolean leaves."""
    import jax

    def leaf(x):
        arr = jnp.asarray(x)
        if jnp.issubdtype(arr.dtype, jnp.inexact):
            return jnp.full(arr.shape, value, dtype=arr.dtype)
        return jnp.zeros(arr.shape, dtype=jax.dtypes.float0)

    return jax.tree_util.tree_map(leaf, response)


def test_fd_lagged_response_jvp_matches_the_primal_identity():
    import jax

    model = _fd_model()
    state0, state1, _ = _fd_states()
    response0 = model.build_lagged_response(state0)

    def build(state, previous):
        return model.build_lagged_response(state, previous_response=previous)

    dstate = jax.tree_util.tree_map(jnp.ones_like, state1)
    dprevious = _response_tangent(response0, 1.0)

    primal, tangent = jax.jvp(build, (state1, response0), (dstate, dprevious))

    assert jnp.allclose(primal.reference_basis, response0.reference_basis)
    assert jnp.allclose(tangent.reference_basis, dprevious.reference_basis)
    assert jnp.allclose(tangent.reference_flux["Q"], dprevious.reference_flux["Q"])


def test_non_fd_lagged_response_jvp_still_tracks_the_state():
    import jax

    model = _fd_model(lagged_response_mode="none")
    state0, state1, _ = _fd_states()
    response0 = model.build_lagged_response(state0)

    def build(state, previous):
        return model.build_lagged_response(state, previous_response=previous)

    dstate = jax.tree_util.tree_map(jnp.ones_like, state1)
    dprevious = _response_tangent(response0, 0.0)

    _, tangent = jax.jvp(build, (state1, response0), (dstate, dprevious))

    assert jnp.allclose(tangent.reference_state.pressure, dstate.pressure)


def test_radau_prepare_lagged_response_rebuild_matches_reuse_for_fd():
    from jax.flatten_util import ravel_pytree

    from NEOPAX._transport_solvers import _radau_prepare_lagged_response

    class KernelContext:
        use_transport_lagged_response = True

    class Carry:
        def __init__(self, y, valid, cache):
            self.y = y
            self.lagged_response_valid = valid
            self.lagged_response_cache = cache
            self.lagged_reference_y = y

    model = _fd_model()
    state0, state1, _ = _fd_states()
    response0 = model.build_lagged_response(state0)
    flat_y1, unravel = ravel_pytree(state1)

    fluxes = {}
    for label, valid in (("reuse", True), ("rebuild", False)):
        response, _, _ = _radau_prepare_lagged_response(
            KernelContext(),
            Carry(flat_y1, jnp.asarray(valid), response0),
            unravel,
            None,
            model.build_lagged_response,
        )
        fluxes[label] = model.evaluate_with_lagged_response(state1, response)["Q"]

    assert jnp.array_equal(fluxes["reuse"], fluxes["rebuild"])
    assert bool(jnp.all(fluxes["rebuild"][0] < model(state=None)["Q"][0]))
