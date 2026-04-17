import jax
import jax.numpy as jnp
from NEOPAX.NEOPAX._sources import SourceModelBase, AnalyticSource, CombinedSourceModel, register_source, get_source
import pytest
from NEOPAX.NEOPAX._source_models import (
    FusionPowerFractionElectronsSource,
    DTReactionSource,
    PowerExchangeSource,
    BremsstrahlungRadiationSource,
)
from NEOPAX.NEOPAX._state import TransportState

class DummySpecies:
        @property
        def species_idx(self):
            return {'e': 0, 'D': 1, 'T': 2, 'He': 3}
    def __init__(self):
        self.mass = jnp.array([1.0, 2.014, 3.016, 4.0])
        self.charge = jnp.array([-1.0, 1.0, 1.0, 2.0])

def make_dummy_state():
    density = jnp.ones((4, 10))
    temperature = jnp.ones((4, 10)) * 10.0
    Er = jnp.zeros(10)
    return TransportState(density=density, temperature=temperature, Er=Er)

def test_analytic_source_array():
    arr = jnp.arange(5.0)
    src = AnalyticSource(arr)
    out = src(None)
    assert jnp.allclose(out, arr)

def test_analytic_source_callable():
    def profile(state):
        return jnp.ones(3) * 2.0
    src = AnalyticSource(profile)
    out = src(None)
    assert jnp.allclose(out, jnp.ones(3) * 2.0)

def test_combined_source_model():
    src1 = AnalyticSource(jnp.ones(3))
    src2 = AnalyticSource(jnp.ones(3) * 2)
    combined = CombinedSourceModel([src1, src2])
    out = combined(None)
    assert jnp.allclose(out, jnp.ones(3) * 3)

def test_register_and_get_source():
    class DummySource(SourceModelBase):
        def __call__(self, state):
            return jnp.ones(2)
    register_source('dummy', DummySource)
    src = get_source('dummy')
    out = src(None)
    assert jnp.allclose(out, jnp.ones(2))

def test_jax_jit_compatibility():
    src = AnalyticSource(jnp.ones(4))
    jitted = jax.jit(src)
    out = jitted(None)
    assert jnp.allclose(out, jnp.ones(4))

def test_fusion_power_fraction_electrons():
    state = make_dummy_state()
    species = DummySpecies()
    src = FusionPowerFractionElectronsSource()
    result = src(state, species)
    assert result['fusion_power_fraction_electrons'].shape == state.temperature[species.species_idx['e']].shape

def test_dt_reaction():
    state = make_dummy_state()
    species = DummySpecies()
    src = DTReactionSource()
    result = src(state, species)
    assert result['DTreactionRate'].shape == state.temperature[species.species_idx['T']].shape

def test_power_exchange():
    state = make_dummy_state()
    species = DummySpecies()
    src = PowerExchangeSource(idx_a='D', idx_b='T')
    result = src(state, species)
    assert result['power_exchange'].shape == state.temperature[species.species_idx['D']].shape

def test_bremsstrahlung_radiation():
    state = make_dummy_state()
    species = DummySpecies()
    src = BremsstrahlungRadiationSource()
    result = src(state, species)
    assert result['PBrems'].shape == state.temperature[species.species_idx['e']].shape
    assert result['Zeff'].shape == state.temperature[species.species_idx['e']].shape
