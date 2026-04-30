import dataclasses
from types import SimpleNamespace

import jax.numpy as jnp

import NEOPAX.main as main_module
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

    def fake_build_transport_flux_model(neo, turb, classical, include_turbulent_particle_flux=True):
        return {
            "neo": neo,
            "turb": turb,
            "classical": classical,
            "include_turbulent_particle_flux": include_turbulent_particle_flux,
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
