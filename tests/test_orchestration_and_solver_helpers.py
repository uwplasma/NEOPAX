import dataclasses
from types import SimpleNamespace

import jax
import jax.numpy as jnp

import NEOPAX._orchestrator as main_module
import NEOPAX._transport_equations as transport_equations_module
import NEOPAX._transport_flux_models as flux_models_module
from NEOPAX._boundary_conditions import BoundaryConditionModel
from NEOPAX._state import TransportState
from NEOPAX._transport_equations import ComposedEquationSystem
from NEOPAX._transport_solvers import (
    _pack_transport_state_arrays,
    _project_state_to_quasi_neutrality,
    _unpack_transport_state_arrays,
)


@dataclasses.dataclass(frozen=True)
class DummySpecies:
    number_species: int
    names: tuple[str, ...]
    charge_qp: jnp.ndarray
    ion_indices: tuple[int, ...]
    species_idx: dict[str, int]


def _dummy_state():
    density = jnp.array(
        [
            [0.0, 0.0],
            [2.0, 3.0],
            [4.0, 5.0],
        ]
    )
    pressure = jnp.array(
        [
            [10.0, 10.0],
            [20.0, 30.0],
            [40.0, 50.0],
        ]
    )
    return TransportState(density=density, pressure=pressure, Er=jnp.array([1.0, 2.0]))


def _dummy_species():
    return DummySpecies(
        number_species=3,
        names=("e", "D", "T"),
        charge_qp=jnp.array([-1.0, 1.0, 1.0]),
        ion_indices=(1, 2),
        species_idx={"e": 0, "D": 1, "T": 2},
    )


def test_build_flux_model_passes_boundary_conditions_and_particle_flux_toggle(monkeypatch):
    species = _dummy_species()
    geometry = SimpleNamespace(dr=0.25)
    config = {
        "neoclassical": {"flux_model": "neo_model", "collisionality_model": "full"},
        "turbulence": {"flux_model": "none"},
        "classical": {"flux_model": "none"},
        "boundary": {
            "density": {"left": {"type": "dirichlet", "value": {"e": 1.0, "D": 2.0, "T": 3.0}}},
            "temperature": {"right": {"type": "neumann", "gradient": {"default": 0.0}}},
        },
        "transport_solver": {"include_turbulent_particle_flux": False},
    }

    factory_calls = []

    def fake_get_transport_flux_model(name):
        def factory(*args, **kwargs):
            factory_calls.append((name, args, kwargs))
            return f"{name}_instance"

        return factory

    def fake_build_transport_flux_model(neo, turb, classical, include_turbulent_particle_flux=True, **kwargs):
        return {
            "neo": neo,
            "turb": turb,
            "classical": classical,
            "include_turbulent_particle_flux": include_turbulent_particle_flux,
            **kwargs,
        }

    monkeypatch.setattr(main_module, "get_transport_flux_model", fake_get_transport_flux_model)
    monkeypatch.setattr(main_module, "build_transport_flux_model", fake_build_transport_flux_model)

    out = main_module._build_flux_model(
        config,
        species=species,
        energy_grid="grid",
        geometry=geometry,
        database="db",
        source_models=None,
    )

    assert out["include_turbulent_particle_flux"] is False
    assert out["center_flux_mode"] == "direct"
    assert out["neo"] == "neo_model_instance"
    assert out["turb"] == "none_instance"
    assert out["classical"] == "none_instance"

    neo_call = next(call for call in factory_calls if call[0] == "neo_model")
    assert neo_call[2]["collisionality_model"] == "full"
    assert neo_call[2]["bc_density"] is not None
    assert neo_call[2]["bc_temperature"] is not None


def test_calculate_sources_from_config_uses_provided_source_models():
    species = _dummy_species()
    state = _dummy_state()

    source_models = {
        "density": lambda s: {"density_source": jnp.ones_like(s.density)},
        "temperature": lambda s: {"pressure_source": 2.0 * jnp.ones_like(s.pressure)},
    }
    params = {"species": species}
    config = {
        "sources": {
            "sources_plot": True,
            "sources_write_hdf5": True,
            "sources_output_dir": "./outputs/unit_sources",
        }
    }

    sources, do_plot, do_hdf5, output_dir = main_module.calculate_sources_from_config(
        state,
        config,
        params,
        source_models=source_models,
    )

    assert do_plot is True
    assert do_hdf5 is True
    assert output_dir == "./outputs/unit_sources"
    assert "density_components" in sources
    assert "pressure_components" in sources
    assert sources["density_total"].shape == state.density.shape
    assert sources["pressure_total"].shape == state.pressure.shape


def test_composed_equation_system_expands_reduced_density_rhs_and_zeroes_electron_row():
    species = _dummy_species()
    state = _dummy_state()

    class DummyDensityEq:
        name = "density"

        def __call__(self, working_state, fluxes=None):
            del working_state, fluxes
            return jnp.array([[7.0, 8.0], [9.0, 10.0]])

    class DummyTemperatureEq:
        name = "temperature"

        def __call__(self, working_state, fluxes=None):
            del fluxes
            return 2.0 * jnp.ones_like(working_state.pressure)

    class DummyErEq:
        name = "Er"

        def __call__(self, working_state, fluxes=None):
            del fluxes
            return 3.0 * jnp.ones_like(working_state.Er)

    eq_system = ComposedEquationSystem(
        equations=(DummyDensityEq(), DummyTemperatureEq(), DummyErEq()),
        density_equation=DummyDensityEq(),
        temperature_equation=DummyTemperatureEq(),
        er_equation=DummyErEq(),
        species=species,
        shared_flux_model=lambda working_state: {"marker": jnp.sum(working_state.Er)},
    )

    rhs = eq_system(0.0, state, runtime=None)
    assert jnp.allclose(rhs.density, jnp.array([[0.0, 0.0], [7.0, 8.0], [9.0, 10.0]]))
    assert jnp.allclose(rhs.pressure, 2.0 * jnp.ones_like(state.pressure))
    assert jnp.allclose(rhs.Er, 3.0 * jnp.ones_like(state.Er))


def test_extract_right_constraints_handles_bc_types():
    state_arr = jnp.array([[1.0, 2.0], [3.0, 4.0]])

    rv, rg = flux_models_module._extract_right_constraints(None, state_arr)
    assert jnp.allclose(rv, jnp.array([2.0, 4.0]))
    assert jnp.allclose(rg, jnp.zeros(2))

    bc_neumann = BoundaryConditionModel(dr=1.0, right_type="neumann", right_gradient=jnp.array([0.5, -0.5]))
    rv, rg = flux_models_module._extract_right_constraints(bc_neumann, state_arr)
    assert jnp.allclose(rv, jnp.array([2.0, 4.0]))
    assert jnp.allclose(rg, jnp.array([0.5, -0.5]))

    bc_robin = BoundaryConditionModel(dr=1.0, right_type="robin", right_decay_length=jnp.array([2.0, 4.0]))
    rv, rg = flux_models_module._extract_right_constraints(bc_robin, state_arr)
    assert jnp.allclose(rv, jnp.array([2.0, 4.0]))
    assert jnp.allclose(rg, jnp.array([-1.0, -1.0]))


def test_ntx_local_particle_flux_evaluator_passes_bc_constraints(monkeypatch):
    species = _dummy_species()
    geometry = SimpleNamespace()
    state = _dummy_state()

    captured = {}

    def fake_get_neoclassical_fluxes(
        species_arg,
        energy_grid_arg,
        geometry_arg,
        database_arg,
        er_profile,
        temperature,
        density,
        **kwargs,
    ):
        captured["kwargs"] = kwargs
        captured["er_profile"] = er_profile
        gamma = jnp.stack([er_profile, er_profile + 1.0, er_profile + 2.0], axis=0)
        q = jnp.zeros_like(gamma)
        upar = jnp.zeros_like(gamma)
        return None, gamma, q, upar

    monkeypatch.setattr(flux_models_module, "get_Neoclassical_Fluxes", fake_get_neoclassical_fluxes)

    bc_density = BoundaryConditionModel(dr=1.0, right_type="neumann", right_gradient=jnp.array([0.1, 0.2, 0.3]))
    bc_temperature = BoundaryConditionModel(dr=1.0, right_type="dirichlet", right_value=jnp.array([5.0, 6.0, 7.0]))
    model = flux_models_module.NTXDatabaseTransportModel(
        species=species,
        energy_grid="grid",
        geometry=geometry,
        database="db",
        bc_density=bc_density,
        bc_temperature=bc_temperature,
    )

    evaluator = model.build_local_particle_flux_evaluator(state)
    out = evaluator(1, 9.0)

    assert jnp.allclose(out, jnp.array([9.0, 10.0, 11.0]))
    assert jnp.allclose(captured["kwargs"]["density_right_grad_constraint"], jnp.array([0.1, 0.2, 0.3]))
    assert jnp.allclose(captured["kwargs"]["temperature_right_constraint"], jnp.array([5.0, 6.0, 7.0]))
    assert float(captured["er_profile"][1]) == 9.0


def test_ntx_database_lagged_face_response_matches_reference_and_finite_difference(monkeypatch):
    """The database lagged response preserves direct centres and face JVPs.

    This uses a nonlinear stand-in for the database interpolation so that the
    finite-difference comparison exercises density, temperature, Er, and face
    gradient dependence rather than passing trivially for a linear flux.
    """
    species = _dummy_species()
    geometry = SimpleNamespace(
        r_grid_half=jnp.asarray([0.0, 0.25, 0.65, 1.0]),
        r_grid=jnp.asarray([0.125, 0.45, 0.825]),
    )
    state0 = TransportState(
        density=jnp.asarray(
            [[1.2, 1.4, 1.7], [0.8, 0.9, 1.1], [0.5, 0.6, 0.75]]
        ),
        pressure=jnp.asarray(
            [[2.4, 3.08, 4.25], [1.04, 1.26, 1.65], [0.45, 0.66, 0.975]]
        ),
        Er=jnp.asarray([0.15, 0.22, 0.31]),
    )
    direction = TransportState(
        density=jnp.asarray(
            [[0.06, -0.03, 0.04], [-0.02, 0.05, -0.01], [0.03, 0.01, -0.02]]
        ),
        pressure=jnp.asarray(
            [[0.08, -0.04, 0.05], [-0.03, 0.06, -0.02], [0.02, 0.03, -0.01]]
        ),
        Er=jnp.asarray([0.02, -0.01, 0.03]),
    )

    def fake_face_database_fluxes(
        species_arg,
        energy_grid_arg,
        geometry_arg,
        database_arg,
        er_faces,
        temperature_faces,
        density_faces,
        dndr_faces,
        dtdr_faces,
        **kwargs,
    ):
        del species_arg, energy_grid_arg, geometry_arg, database_arg, kwargs
        er_by_species = er_faces[None, :]
        gamma = (
            density_faces * er_by_species
            + 0.3 * dndr_faces
            + 0.1 * temperature_faces**2
        )
        heat = density_faces * temperature_faces**2 + 0.2 * dtdr_faces * er_by_species
        upar = er_by_species**2 + 0.4 * density_faces * dtdr_faces
        return None, gamma, heat, upar

    def fake_center_database_fluxes(
        species_arg,
        energy_grid_arg,
        geometry_arg,
        database_arg,
        er,
        temperature,
        density,
        **kwargs,
    ):
        del species_arg, energy_grid_arg, geometry_arg, database_arg, kwargs
        er_by_species = er[None, :]
        gamma = density * er_by_species + 0.1 * temperature**2
        heat = density * temperature**2 + 0.2 * er_by_species**2
        upar = er_by_species * temperature + 0.3 * density**2
        return None, gamma, heat, upar

    monkeypatch.setattr(
        flux_models_module,
        "get_Neoclassical_Fluxes_Faces",
        fake_face_database_fluxes,
    )
    monkeypatch.setattr(
        flux_models_module,
        "get_Neoclassical_Fluxes",
        fake_center_database_fluxes,
    )
    model = flux_models_module.NTXDatabaseTransportModel(
        species=species,
        energy_grid="grid",
        geometry=geometry,
        database="database",
    )

    def face_fluxes_from_state(state):
        face_state = flux_models_module.build_face_transport_state(state, geometry)
        return model.evaluate_face_fluxes(state, face_state)

    response = model.build_lagged_response(state0)
    lagged_at_reference = model.evaluate_with_lagged_response(state0, response)
    direct_at_reference = face_fluxes_from_state(state0)
    for name in ("Gamma", "Q", "Upar"):
        assert jnp.allclose(
            lagged_at_reference[name],
            model(state0)[name],
            rtol=1.0e-6,
            atol=1.0e-6,
        )
        assert jnp.allclose(
            lagged_at_reference[f"{name}_faces"],
            direct_at_reference[name],
            rtol=1.0e-6,
            atol=1.0e-6,
        )

    epsilon = jnp.asarray(1.0e-3)
    state_plus = jax.tree_util.tree_map(
        lambda value, delta: value + epsilon * delta,
        state0,
        direction,
    )
    state_minus = jax.tree_util.tree_map(
        lambda value, delta: value - epsilon * delta,
        state0,
        direction,
    )
    finite_difference = jax.tree_util.tree_map(
        lambda plus, minus: (plus - minus) / (2.0 * epsilon),
        face_fluxes_from_state(state_plus),
        face_fluxes_from_state(state_minus),
    )
    lagged_direction = jax.tree_util.tree_map(
        lambda value, reference: value - reference,
        model.evaluate_with_lagged_response(state_plus, response),
        lagged_at_reference,
    )
    center_finite_difference = jax.tree_util.tree_map(
        lambda plus, minus: (plus - minus) / (2.0 * epsilon),
        model(state_plus),
        model(state_minus),
    )
    for name in ("Gamma", "Q", "Upar"):
        assert jnp.allclose(
            lagged_direction[name] / epsilon,
            center_finite_difference[name],
            rtol=3.0e-3,
            atol=3.0e-4,
        )
        assert jnp.allclose(
            lagged_direction[f"{name}_faces"] / epsilon,
            finite_difference[name],
            rtol=3.0e-3,
            atol=3.0e-4,
        )


def test_pack_and_unpack_transport_state_arrays_restore_electron_row():
    species = _dummy_species()
    state = _dummy_state()

    packed = _pack_transport_state_arrays(state, species)
    assert packed[0].shape == (2, 2)

    unpacked = _unpack_transport_state_arrays(
        packed,
        state,
        species=species,
        temperature_active_mask=jnp.array([True, True, True]),
        fixed_temperature_profile=state.temperature,
        density_floor=1.0e-6,
        temperature_floor=None,
    )
    assert unpacked.density.shape == state.density.shape
    assert jnp.allclose(unpacked.density[1:], state.density[1:])
    assert jnp.allclose(unpacked.density[0], state.density[1] + state.density[2])


LAGGED_HEAT_FLUX = 7.0


class _SingleTemperatureEquation:
    name = "temperature"

    def __init__(self, flux_model=None):
        self.flux_model = flux_model

    def __call__(self, working_state, fluxes=None):
        if fluxes is None:
            fluxes = self.flux_model(working_state)
        return fluxes["Q"]


class _LaggedFluxModel:
    """Flux model whose lagged response returns a heat flux the direct call never produces."""

    def __call__(self, state):
        return {
            "Gamma": jnp.zeros_like(state.density),
            "Q": jnp.zeros_like(state.pressure),
            "Upar": jnp.zeros_like(state.density),
        }

    def build_lagged_response(self, state, **kwargs):
        del kwargs
        return {"reference_pressure": state.pressure}

    def evaluate_with_lagged_response(self, state, lagged_response, **kwargs):
        del lagged_response, kwargs
        return {
            "Gamma": jnp.zeros_like(state.density),
            "Q": jnp.full_like(state.pressure, LAGGED_HEAT_FLUX),
            "Upar": jnp.zeros_like(state.density),
        }


def test_single_equation_solve_applies_the_lagged_flux_response(monkeypatch):
    state = _dummy_state()
    flux_model = _LaggedFluxModel()
    runtime = main_module.RuntimeContext(
        species=_dummy_species(),
        energy_grid=None,
        geometry=SimpleNamespace(dr=0.25),
        database=None,
        solver_parameters={"t0": 0.0, "t_final": 1.0, "dt": 0.1, "rtol": 1.0e-6, "atol": 1.0e-8},
        models=main_module.Models(flux=flux_model, source={}),
    )
    monkeypatch.setattr(
        transport_equations_module,
        "build_equation_system",
        lambda **kwargs: [_SingleTemperatureEquation(flux_model)],
    )

    prepared = main_module.prepare_transport_solver_components({}, runtime, state)
    equation_system = prepared["equation_system"]

    assert len(prepared["equations_to_evolve"]) == 1
    assert equation_system.shared_flux_model is flux_model

    # The solver drives exactly this pair via _lagged_response_hooks.
    lagged = equation_system.build_lagged_response(state)
    assert lagged.flux_response is not None

    rhs = equation_system.evaluate_with_lagged_response(0.0, state, None, lagged)
    assert jnp.allclose(rhs.pressure, LAGGED_HEAT_FLUX)


def test_with_geometry_payload_keeps_the_shared_flux_model_for_one_equation(monkeypatch):
    flux_model = object()
    equation_system = ComposedEquationSystem(
        equations=(_SingleTemperatureEquation(),),
        temperature_equation=_SingleTemperatureEquation(),
        species=_dummy_species(),
        shared_flux_model=flux_model,
        config={},
        solver_cfg={},
        boundary_models={},
    )
    monkeypatch.setattr(
        transport_equations_module,
        "build_equation_system",
        lambda **kwargs: [_SingleTemperatureEquation()],
    )

    rebuilt = equation_system.with_geometry_payload(SimpleNamespace(dr=0.25))

    assert len(rebuilt.equations) == 1
    assert rebuilt.shared_flux_model is flux_model


def test_project_state_to_quasi_neutrality_and_fixed_temperature_projection():
    species = _dummy_species()
    state = _dummy_state()
    fixed_temperature = jnp.array([[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]])

    out = _project_state_to_quasi_neutrality(
        state,
        species,
        temperature_active_mask=jnp.array([True, False, True]),
        fixed_temperature_profile=fixed_temperature,
        density_floor=1.0e-6,
        temperature_floor=None,
    )

    assert jnp.allclose(out.density[0], state.density[1] + state.density[2])
    assert jnp.allclose(out.temperature[1], jnp.array([5.0, 5.0]))
