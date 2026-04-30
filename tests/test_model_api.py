import dataclasses

import jax.numpy as jnp
import pytest

from NEOPAX._model_api import ModelCapabilities, ModelValidationContext
from NEOPAX._source_models import SourceModelBase, get_source_model, register_source_model
from NEOPAX._state import TransportState
from NEOPAX._transport_flux_models import (
    TransportFluxModelBase,
    get_transport_flux_model,
    get_transport_flux_model_capabilities,
    register_transport_flux_model,
)
from NEOPAX import make_validation_context


@dataclasses.dataclass(frozen=True)
class DummySpecies:
    number_species: int = 2
    names: tuple[str, ...] = ("e", "D")
    charge_qp: jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.array([-1.0, 1.0]))


def _state():
    density = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    pressure = 2.0 * density
    return TransportState(density=density, pressure=pressure, Er=jnp.array([0.1, 0.2, 0.3]))


def _face_state():
    density = jnp.array([[1.0, 2.0, 3.0, 4.0], [4.0, 5.0, 6.0, 7.0]])
    pressure = 2.0 * density
    return TransportState(density=density, pressure=pressure, Er=jnp.array([0.0, 0.1, 0.2, 0.3]))


@dataclasses.dataclass(frozen=True, eq=False)
class GoodFluxModel(TransportFluxModelBase):
    scale: float = 1.0

    def __call__(self, state, geometry=None, params=None):
        del geometry, params
        base = self.scale * jnp.ones_like(state.density)
        return {"Gamma": base, "Q": 2.0 * base, "Upar": jnp.zeros_like(base)}

    def build_local_particle_flux_evaluator(self, state):
        gamma = self(state)["Gamma"]

        def evaluator(radius_index, er_value):
            del er_value
            return gamma[:, radius_index]

        return evaluator

    def evaluate_face_fluxes(self, state, face_state, **kwargs):
        del state, kwargs
        base = self.scale * jnp.ones_like(face_state.density)
        return {"Gamma": base, "Q": 2.0 * base, "Upar": jnp.zeros_like(base)}


@dataclasses.dataclass(frozen=True, eq=False)
class BadFluxModel(TransportFluxModelBase):
    def __call__(self, state, geometry=None, params=None):
        del state, geometry, params
        return {"Gamma": jnp.array([1.0])}


@dataclasses.dataclass(frozen=True, eq=False)
class GoodSourceModel(SourceModelBase):
    amplitude: float = 1.0

    def __call__(self, state):
        return {"pressure_source": self.amplitude * jnp.ones_like(state.pressure)}


@dataclasses.dataclass(frozen=True, eq=False)
class BadSourceModel(SourceModelBase):
    def __call__(self, state):
        del state
        return {"bad_source": object()}


def test_register_transport_flux_model_with_validation():
    state = _state()
    context = make_validation_context(
        builder_kwargs={"scale": 3.0},
        n_species=state.density.shape[0],
        n_radial=state.density.shape[1],
    )
    context = dataclasses.replace(context, state=state, face_state=_face_state())
    register_transport_flux_model(
        "unit_test_flux_good",
        GoodFluxModel,
        capabilities=ModelCapabilities(
            jit_safe=True,
            autodiff_safe=True,
            vmap_safe=True,
            local_evaluator=True,
            face_fluxes=True,
        ),
        validate=True,
        validation_context=context,
    )

    model = get_transport_flux_model("unit_test_flux_good")(scale=2.0)
    out = model(state)
    assert out["Gamma"].shape == state.density.shape
    assert get_transport_flux_model_capabilities("unit_test_flux_good").local_evaluator is True
    assert get_transport_flux_model_capabilities("unit_test_flux_good").autodiff_safe is True


def test_register_transport_flux_model_validation_rejects_bad_output():
    with pytest.raises((KeyError, ValueError, TypeError)):
        register_transport_flux_model(
            "unit_test_flux_bad",
            BadFluxModel,
            validate=True,
            validation_context=make_validation_context(builder_kwargs={}, n_species=2, n_radial=3),
        )


def test_register_source_model_with_validation():
    context = make_validation_context(builder_kwargs={"amplitude": 5.0}, n_species=2, n_radial=3)
    register_source_model(
        "unit_test_source_good",
        GoodSourceModel,
        validate=True,
        validation_context=context,
    )
    model = get_source_model("unit_test_source_good", amplitude=2.0)
    out = model(_state())
    assert out["pressure_source"].shape == _state().pressure.shape


def test_register_source_model_validation_rejects_bad_output():
    with pytest.raises((TypeError, ValueError)):
        register_source_model(
            "unit_test_source_bad",
            BadSourceModel,
            validate=True,
            validation_context=make_validation_context(builder_kwargs={}, n_species=2, n_radial=3),
        )


def test_make_validation_context_builds_default_face_state():
    context = make_validation_context(builder_kwargs={"x": 1}, n_species=3, n_radial=5)
    assert context.builder_kwargs == {"x": 1}
    assert context.state.density.shape == (3, 5)
    assert context.state.pressure.shape == (3, 5)
    assert context.state.Er.shape == (5,)
    assert context.face_state is not None
    assert context.face_state.density.shape == (3, 6)
