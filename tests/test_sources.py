import jax
import jax.numpy as jnp
from NEOPAX._source_models import (
    AnalyticSource,
    BremsstrahlungRadiationSource,
    CombinedSourceModel,
    DTReactionSource,
    FusionPowerFractionElectronsSource,
    PowerExchangeSource,
    SourceModelBase,
    get_source,
    register_source,
)
from NEOPAX._species import Species
from NEOPAX._state import TransportState


def make_dummy_state():
    density = jnp.ones((4, 10))
    temperature = jnp.ones((4, 10)) * 10.0
    pressure = density * temperature
    Er = jnp.zeros(10)
    return TransportState(density=density, pressure=pressure, Er=Er)


def make_dummy_species():
    return Species(
        number_species=4,
        species_indices=jnp.arange(4),
        mass_mp=jnp.array([1.0 / 1836.15267343, 2.014, 3.016, 4.0]),
        charge_qp=jnp.array([-1.0, 1.0, 1.0, 2.0]),
        names=("e", "D", "T", "He"),
    )


def test_analytic_source_array():
    arr = jnp.arange(5.0)
    src = AnalyticSource(arr)
    out = src(None)
    assert jnp.allclose(out, arr)


def test_analytic_source_callable():
    def profile(state):
        del state
        return jnp.ones(3) * 2.0

    src = AnalyticSource(profile)
    out = src(None)
    assert jnp.allclose(out, jnp.ones(3) * 2.0)


def test_combined_source_model():
    class DictSource(SourceModelBase):
        def __init__(self, value):
            self.value = value

        def __call__(self, state):
            del state
            return {"src": jnp.asarray(self.value)}

    src1 = DictSource(jnp.ones(3))
    src2 = DictSource(jnp.ones(3) * 2)
    combined = CombinedSourceModel((src1, src2))
    out = combined(None)
    assert "src" in out
    assert jnp.allclose(out["src"], jnp.ones(3) * 3)


def test_combined_source_model_with_added_sources():
    class DictSource(SourceModelBase):
        def __init__(self, value):
            self.value = value

        def __call__(self, state):
            del state
            return {"src": jnp.asarray(self.value)}

    base = CombinedSourceModel((DictSource(jnp.ones(3)),))
    updated = base.with_added_sources(DictSource(jnp.ones(3) * 2))

    assert len(base.sources) == 1
    assert len(updated.sources) == 2
    assert jnp.allclose(updated(None)["src"], jnp.ones(3) * 3)


def test_combined_source_model_with_replaced_sources():
    class DictSource(SourceModelBase):
        def __init__(self, value):
            self.value = value

        def __call__(self, state):
            del state
            return {"src": jnp.asarray(self.value)}

    base = CombinedSourceModel((DictSource(jnp.ones(3)),))
    updated = base.with_replaced_sources((DictSource(jnp.ones(3) * 5),))

    assert len(updated.sources) == 1
    assert jnp.allclose(updated(None)["src"], jnp.ones(3) * 5)


def test_register_and_get_source():
    class DummySource(SourceModelBase):
        def __call__(self, state):
            del state
            return jnp.ones(2)

    register_source("dummy_unit_source", DummySource)
    src = get_source("dummy_unit_source")
    out = src(None)
    assert jnp.allclose(out, jnp.ones(2))


def test_jax_jit_compatibility():
    src = AnalyticSource(jnp.ones(4))
    jitted = jax.jit(src)
    out = jitted(None)
    assert jnp.allclose(out, jnp.ones(4))


def test_fusion_power_fraction_electrons():
    state = make_dummy_state()
    species = make_dummy_species()
    src = FusionPowerFractionElectronsSource()
    result = src(state, species)
    assert result["fusion_power_fraction_electrons"].shape == state.temperature[species.species_idx["e"]].shape


def test_dt_reaction():
    state = make_dummy_state()
    species = make_dummy_species()
    src = DTReactionSource()
    result = src(state, species)
    assert result["DTreactionRate"].shape == state.temperature[species.species_idx["T"]].shape
    assert result["HeSource"].shape == state.temperature[species.species_idx["T"]].shape
    assert result["AlphaPower"].shape == state.temperature[species.species_idx["T"]].shape


def test_power_exchange():
    state = make_dummy_state()
    species = make_dummy_species()
    src = PowerExchangeSource(mode="all")
    result = src(state, species)
    assert result["power_exchange"].shape == state.temperature.shape


def test_bremsstrahlung_radiation():
    state = make_dummy_state()
    species = make_dummy_species()
    src = BremsstrahlungRadiationSource()
    result = src(state, species)
    assert result["PBrems"].shape == state.temperature[species.species_idx["e"]].shape
    assert result["Zeff"].shape == state.temperature[species.species_idx["e"]].shape
