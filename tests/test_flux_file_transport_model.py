import contextlib
import io
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import pytest

from NEOPAX._entropy_models import get_entropy_model
from NEOPAX._transport_flux_models import (
    AnalyticalTurbulentTransportModel,
    CombinedTransportFluxModel,
    CombinedTransportLaggedResponse,
    FluxesRFileTransportModel,
    PowerAnalyticalTurbulentTransportModel,
    SpectraXTurbulenceFDLaggedResponse,
    ReLUAnalyticalTurbulentTransportModel,
    _sum_float_delta_bar_trees,
    build_fluxes_r_file_transport_model,
    read_flux_profile_file,
)
from NEOPAX._fem import cell_centered_from_faces, faces_from_cell_centered
from NEOPAX._orchestrator import calculate_fluxes_from_config
from NEOPAX._state import TransportState


def test_sum_float_delta_bar_trees_converts_prepared_static_float0_leaves():
    """Joint prepared-support paths retain mapped bar axes for static leaves."""
    primal = {
        "coefficient": jnp.asarray([2.0, -1.0]),
        "mode_index": jnp.asarray([1, 3], dtype=jnp.int32),
    }
    # The energy-map axis is present on every cotangent leaf, including the
    # float0 cotangent of an integer/static prepared-system leaf.
    float0_index_bar = jnp.zeros((2, 2), dtype=jax.dtypes.float0)
    total = _sum_float_delta_bar_trees(
        primal,
        {"coefficient": jnp.asarray([[1.0, 2.0], [0.0, 1.0]]), "mode_index": float0_index_bar},
        {"coefficient": jnp.asarray([[-0.5, 3.0], [2.0, 0.0]]), "mode_index": float0_index_bar},
        {"coefficient": jnp.asarray([[0.25, -1.0], [1.0, -2.0]]), "mode_index": float0_index_bar},
        {"coefficient": jnp.asarray([[0.0, 0.5], [0.0, 1.0]]), "mode_index": float0_index_bar},
    )
    assert jnp.allclose(total["coefficient"], jnp.asarray([[0.75, 4.5], [3.0, 0.0]]))
    assert total["mode_index"].shape == (2, 2)
    assert total["mode_index"].dtype == jnp.float64
    assert jnp.allclose(total["mode_index"], jnp.zeros((2, 2)))


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


def _write_flux_file(
    path: Path,
    r,
    gamma=None,
    q=None,
    upar=None,
    *,
    r_center=None,
    gamma_center=None,
    q_center=None,
    upar_center=None,
):
    with h5py.File(path, "w") as f:
        f["r"] = jnp.asarray(r)
        if gamma is not None:
            f["Gamma"] = jnp.asarray(gamma)
        if q is not None:
            f["Q"] = jnp.asarray(q)
        if upar is not None:
            f["Upar"] = jnp.asarray(upar)
        if r_center is not None:
            f["r_center"] = jnp.asarray(r_center)
        if gamma_center is not None:
            f["Gamma_center"] = jnp.asarray(gamma_center)
        if q_center is not None:
            f["Q_center"] = jnp.asarray(q_center)
        if upar_center is not None:
            f["Upar_center"] = jnp.asarray(upar_center)


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


def test_fluxes_r_file_default_is_face_centered_and_reconstructs_cells(tmp_path):
    path = tmp_path / "default_face_fluxes.h5"
    gamma_faces = jnp.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
    q_faces = jnp.array([[10.0, 30.0, 50.0], [20.0, 40.0, 60.0]])
    _write_flux_file(path, r=[0.0, 0.5, 1.0], gamma=gamma_faces, q=q_faces)

    with contextlib.redirect_stdout(io.StringIO()):
        model = build_fluxes_r_file_transport_model(
            DummySpecies(),
            DummyGeometry(),
            fluxes_file=path,
        )

    center_fluxes = model(state=None)
    face_fluxes = model.evaluate_face_fluxes(state=None, face_state=None)

    assert jnp.allclose(face_fluxes["Gamma"], gamma_faces)
    assert jnp.allclose(face_fluxes["Q"], q_faces)
    assert jnp.allclose(center_fluxes["Gamma"], jnp.vstack([cell_centered_from_faces(gamma_faces[0]), cell_centered_from_faces(gamma_faces[1])]))
    assert jnp.allclose(center_fluxes["Q"], jnp.vstack([cell_centered_from_faces(q_faces[0]), cell_centered_from_faces(q_faces[1])]))
    assert jnp.allclose(center_fluxes["Gamma_faces"], gamma_faces)
    assert jnp.allclose(center_fluxes["Q_faces"], q_faces)


def test_fluxes_r_file_default_rejects_center_only_grid_that_misses_boundary_faces(tmp_path):
    path = tmp_path / "center_only_fluxes.h5"
    gamma = jnp.array([[1.0, 3.0], [2.0, 4.0]])
    q = jnp.array([[10.0, 30.0], [20.0, 40.0]])
    _write_flux_file(path, r=[0.25, 0.75], gamma=gamma, q=q)

    with contextlib.redirect_stdout(io.StringIO()):
        with pytest.raises(ValueError, match="does not cover"):
            build_fluxes_r_file_transport_model(
                DummySpecies(),
                DummyGeometry(),
                fluxes_file=path,
            )


def test_fluxes_r_file_rejects_a_file_that_does_not_span_the_cell_grid(tmp_path):
    path = tmp_path / "narrow_cell_fluxes.h5"
    _write_flux_file(path, r=[0.3, 0.7], gamma=jnp.ones((2, 2)), q=jnp.ones((2, 2)))

    with contextlib.redirect_stdout(io.StringIO()):
        with pytest.raises(ValueError, match=r"does not cover geometry\.r_grid"):
            build_fluxes_r_file_transport_model(
                DummySpecies(),
                DummyGeometry(),
                fluxes_file=path,
                grid_location="cell_centered",
            )


def test_fluxes_r_file_rejects_a_file_that_does_not_span_the_face_grid(tmp_path):
    path = tmp_path / "narrow_face_fluxes.h5"
    _write_flux_file(path, r=[0.0, 0.5, 0.9], gamma=jnp.ones((2, 3)), q=jnp.ones((2, 3)))

    with contextlib.redirect_stdout(io.StringIO()):
        with pytest.raises(ValueError, match=r"does not cover geometry\.r_grid_half"):
            build_fluxes_r_file_transport_model(
                DummySpecies(),
                DummyGeometry(),
                fluxes_file=path,
                grid_location="face_centered",
            )


def test_fluxes_r_file_rejects_a_file_grid_too_short_to_interpolate(tmp_path):
    path = tmp_path / "single_point_fluxes.h5"
    _write_flux_file(path, r=[0.5], gamma=jnp.ones((2, 1)), q=jnp.ones((2, 1)))

    with contextlib.redirect_stdout(io.StringIO()):
        with pytest.raises(ValueError, match=r"at least 2"):
            build_fluxes_r_file_transport_model(
                DummySpecies(),
                DummyGeometry(),
                fluxes_file=path,
                grid_location="cell_centered",
            )


def test_fluxes_r_file_rejects_a_descending_file_grid(tmp_path):
    path = tmp_path / "descending_fluxes.h5"
    _write_flux_file(path, r=[0.75, 0.25], gamma=jnp.ones((2, 2)), q=jnp.ones((2, 2)))

    with contextlib.redirect_stdout(io.StringIO()):
        with pytest.raises(ValueError, match=r"strictly increasing"):
            build_fluxes_r_file_transport_model(
                DummySpecies(),
                DummyGeometry(),
                fluxes_file=path,
                grid_location="cell_centered",
            )


def test_fluxes_r_file_rejects_a_file_grid_with_non_finite_radii(tmp_path):
    path = tmp_path / "nonfinite_fluxes.h5"
    _write_flux_file(path, r=[0.25, jnp.nan], gamma=jnp.ones((2, 2)), q=jnp.ones((2, 2)))

    with contextlib.redirect_stdout(io.StringIO()):
        with pytest.raises(ValueError, match=r"non-finite"):
            build_fluxes_r_file_transport_model(
                DummySpecies(),
                DummyGeometry(),
                fluxes_file=path,
                grid_location="cell_centered",
            )


def test_fluxes_r_file_accepts_a_file_whose_endpoints_touch_the_grid(tmp_path):
    path = tmp_path / "touching_fluxes.h5"
    gamma = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    _write_flux_file(path, r=[0.25, 0.5, 0.75], gamma=gamma, q=gamma)

    with contextlib.redirect_stdout(io.StringIO()):
        model = build_fluxes_r_file_transport_model(
            DummySpecies(),
            DummyGeometry(),
            fluxes_file=path,
            grid_location="cell_centered",
        )

    assert jnp.all(jnp.isfinite(model(state=None)["Gamma"]))


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


def test_fluxes_r_file_direct_center_mode_reads_center_datasets(tmp_path):
    path = tmp_path / "face_and_center_fluxes.h5"
    gamma_faces = jnp.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
    q_faces = jnp.array([[10.0, 30.0, 50.0], [20.0, 40.0, 60.0]])
    gamma_center = jnp.array([[11.0, 13.0], [12.0, 14.0]])
    q_center = jnp.array([[110.0, 130.0], [120.0, 140.0]])
    _write_flux_file(
        path,
        r=[0.0, 0.5, 1.0],
        gamma=gamma_faces,
        q=q_faces,
        r_center=[0.25, 0.75],
        gamma_center=gamma_center,
        q_center=q_center,
    )

    with contextlib.redirect_stdout(io.StringIO()):
        model = build_fluxes_r_file_transport_model(
            DummySpecies(),
            DummyGeometry(),
            fluxes_file=path,
            center_flux_mode="file_center",
        )

    center_fluxes = model(state=None)
    face_fluxes = model.evaluate_face_fluxes(state=None, face_state=None)

    assert jnp.allclose(face_fluxes["Gamma"], gamma_faces)
    assert jnp.allclose(face_fluxes["Q"], q_faces)
    assert jnp.allclose(center_fluxes["Gamma"], gamma_center)
    assert jnp.allclose(center_fluxes["Q"], q_center)


def test_fluxes_r_file_direct_center_mode_requires_center_datasets(tmp_path):
    path = tmp_path / "missing_center_fluxes.h5"
    gamma_faces = jnp.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
    _write_flux_file(path, r=[0.0, 0.5, 1.0], gamma=gamma_faces)

    with contextlib.redirect_stdout(io.StringIO()):
        with pytest.warns(RuntimeWarning, match="center_flux_mode='file_center'"):
            with pytest.raises(ValueError, match="requires center-grid datasets"):
                build_fluxes_r_file_transport_model(
                    DummySpecies(),
                    DummyGeometry(),
                    fluxes_file=path,
                    center_flux_mode="file_center",
                )


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


class DummyFDSpecies:
    number_species = 2
    names = ("e", "i")


class DummyFDGeometry:
    def __init__(self):
        self.r_grid_half = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
        self.r_grid = jnp.array([0.125, 0.375, 0.625, 0.875])
        self.dr = 0.25
        self.a_b = 0.5


def _write_fd_flux_file(path: Path, r):
    n_r = len(r)
    gamma = 1.0 + jnp.arange(2 * n_r, dtype=float).reshape(2, n_r)
    q = 10.0 * gamma
    with h5py.File(path, "w") as f:
        f["r"] = jnp.asarray(r)
        f["Gamma"] = gamma
        f["Q"] = q
        f["Upar"] = jnp.zeros_like(gamma)
        # One perturbation channel: (n_perturb, n_species, n_radius).
        f["Gamma_perturb"] = (1.05 * gamma)[None, :, :]
        f["Q_perturb"] = (1.05 * q)[None, :, :]
        f["perturb_delta"] = 0.05 * jnp.ones((1, n_r))
        f["perturb_present"] = jnp.ones((1, n_r), dtype=bool)
        f["perturb_kind"] = ["temperature_gradient"]
        f["perturb_species"] = ["i"]


def _fd_state(n_r):
    profile = jnp.linspace(2.0, 1.0, n_r)
    density = jnp.stack([profile, profile])
    temperature = jnp.stack([3.0 * profile, 2.0 * profile])
    return TransportState(density=density, pressure=density * temperature, Er=jnp.zeros(n_r))


def test_fd_lagged_response_rejects_a_flux_file_off_the_geometry_grid(tmp_path):
    path = tmp_path / "fd_mismatched_fluxes.h5"
    _write_fd_flux_file(path, r=[0.1, 0.3, 0.6, 0.9])

    with contextlib.redirect_stdout(io.StringIO()):
        with pytest.raises(ValueError, match="does not cover|primary NEOPAX flux grid"):
            build_fluxes_r_file_transport_model(
                DummyFDSpecies(),
                DummyFDGeometry(),
                fluxes_file=path,
                lagged_response_mode="fd",
            )


def test_fd_lagged_response_rejects_center_length_perturbation_data_on_face_grid(tmp_path):
    geometry = DummyFDGeometry()
    path = tmp_path / "fd_bad_perturb_shape_fluxes.h5"
    _write_fd_flux_file(path, r=geometry.r_grid_half)
    with h5py.File(path, "r+") as f:
        center_n = geometry.r_grid.shape[0]
        del f["Q_perturb"]
        f["Q_perturb"] = jnp.ones((1, 2, center_n))

    with contextlib.redirect_stdout(io.StringIO()):
        with pytest.raises(ValueError, match="Q_perturb/Q_perturbed"):
            build_fluxes_r_file_transport_model(
                DummyFDSpecies(),
                geometry,
                fluxes_file=path,
                lagged_response_mode="fd",
            )


def test_fd_lagged_response_builds_under_jit(tmp_path):
    geometry = DummyFDGeometry()
    path = tmp_path / "fd_fluxes.h5"
    _write_fd_flux_file(path, r=geometry.r_grid_half)

    with contextlib.redirect_stdout(io.StringIO()):
        model = build_fluxes_r_file_transport_model(
            DummyFDSpecies(),
            geometry,
            fluxes_file=path,
            lagged_response_mode="fd",
        )
    state = _fd_state(geometry.r_grid.shape[0])

    eager = model.build_lagged_response(state)
    jitted = jax.jit(model.build_lagged_response)(state)

    assert isinstance(jitted, SpectraXTurbulenceFDLaggedResponse)
    assert jnp.allclose(jitted.reference_flux["Q_faces"], eager.reference_flux["Q_faces"])
    assert jnp.allclose(jitted.reference_basis, eager.reference_basis)
    assert jnp.allclose(jitted.q_perturb, eager.q_perturb)


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


def test_relu_analytical_turbulent_transport_model_thresholds_fluxes():
    species = type(
        "Species",
        (),
        {
            "species_idx": {"e": 0},
            "species_indices": jnp.array([0, 1]),
        },
    )()
    field = DummyGeometry()
    state = TransportState(
        density=jnp.array([[2.0, 1.0], [2.0, 1.0]]),
        pressure=jnp.array([[20.0, 4.0], [20.0, 4.0]]),
        Er=jnp.zeros((2, 2)),
    )

    off_model = ReLUAnalyticalTurbulentTransportModel(
        species=species,
        field=field,
        density_critical_gradient=jnp.array([1.0e30, 1.0e30]),
        temperature_critical_gradient=jnp.array([1.0e30, 1.0e30]),
        density_relu_slope=jnp.array([0.0, 0.0]),
        temperature_relu_slope=jnp.array([0.0, 0.0]),
    )
    on_model = off_model.with_transport_coeffs(
        density_critical_gradient=0.0,
        temperature_critical_gradient=0.0,
        density_relu_slope=1.0,
        temperature_relu_slope=1.0,
    )

    off_fluxes = off_model(state)
    on_fluxes = on_model(state)

    assert jnp.allclose(off_fluxes["Gamma"], 0.0)
    assert jnp.allclose(off_fluxes["Q"], 0.0)
    assert jnp.any(jnp.abs(on_fluxes["Gamma"]) > 0.0)
    assert jnp.any(jnp.abs(on_fluxes["Q"]) > 0.0)


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


def test_combined_joint_lagged_pullback_preserves_submodel_state_bars():
    """The joint NTX hook must not forward BC kwargs or drop combined bars."""

    class LinearLaggedModel:
        def __init__(self, factor, *, joint=False):
            self.factor = factor
            self.joint = joint

        def pullback_build_lagged_response(self, state, response_bar, **kwargs):
            assert kwargs["bc_density"] == "density-bc"
            assert kwargs["bc_temperature"] == "temperature-bc"
            return self.factor * response_bar

        def pullback_build_lagged_response_state_and_support_payload_batched_interpolated_faces(
            self,
            state,
            response_bars,
            support,
        ):
            assert self.joint
            return self.factor * response_bars, 11.0 * response_bars

    model = CombinedTransportFluxModel(
        neoclassical_model=LinearLaggedModel(3.0, joint=True),
        turbulent_model=LinearLaggedModel(5.0),
        classical_model=LinearLaggedModel(7.0),
    )
    bars = jnp.asarray([2.0, -1.0])
    state_bars, support_bars = (
        model.pullback_build_lagged_response_state_and_support_payload_batched_interpolated_faces(
            jnp.asarray(1.0),
            CombinedTransportLaggedResponse(bars, bars, bars),
            jnp.asarray(4.0),
            bc_density="density-bc",
            bc_temperature="temperature-bc",
        )
    )

    assert jnp.allclose(state_bars, 15.0 * bars)
    assert jnp.allclose(support_bars, 11.0 * bars)

    state_bars_without_classical, _ = (
        model.pullback_build_lagged_response_state_and_support_payload_batched_interpolated_faces(
            jnp.asarray(1.0),
            CombinedTransportLaggedResponse(bars, bars, None),
            jnp.asarray(4.0),
            bc_density="density-bc",
            bc_temperature="temperature-bc",
        )
    )
    assert jnp.allclose(state_bars_without_classical, 8.0 * bars)


def test_combined_batched_primal_reuse_support_pullback_delegates_to_neoclassical():
    class Neoclassical:
        def pullback_build_lagged_response_support_payload_batched_interpolated_faces_reuse_local_vjp_primal(
            self, state, response_bars, support
        ):
            return response_bars + support

    model = CombinedTransportFluxModel(
        neoclassical_model=Neoclassical(), turbulent_model=None, classical_model=None
    )
    bars = jnp.asarray([2.0, -1.0])
    result = model.pullback_build_lagged_response_support_payload_batched_interpolated_faces_reuse_local_vjp_primal(
        jnp.asarray(1.0), CombinedTransportLaggedResponse(bars, None, None), jnp.asarray(4.0)
    )
    assert jnp.allclose(result, bars + 4.0)


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
