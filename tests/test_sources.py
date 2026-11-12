import jax
import jax.numpy as jnp
import numpy as np
from NEOPAX._source_models import (
    AnalyticSource,
    BremsstrahlungRadiationSource,
    CombinedSourceModel,
    DTReactionSource,
    FusionPowerFractionElectronsSource,
    PowerExchangeSource,
    TemperatureEquilibrationPowerExchangeSource,
    SourceModelBase,
    get_source,
    register_source,
)
from NEOPAX._sources import bremsstrahlung_radiation_generalized
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


def test_source_model_default_lagged_response_reuses_outputs():
    class DictSource(SourceModelBase):
        def __call__(self, state):
            del state
            return {"src": jnp.asarray([1.0, 2.0])}

    src = DictSource()
    lagged = src.build_lagged_response(None)
    out = src.evaluate_with_lagged_response(None, lagged)
    assert jnp.allclose(out["src"], jnp.asarray([1.0, 2.0]))


def test_combined_source_model_lagged_response_reuses_component_outputs():
    class DictSource(SourceModelBase):
        def __init__(self, value):
            self.value = value

        def __call__(self, state):
            del state
            return {"src": jnp.asarray(self.value)}

    combined = CombinedSourceModel(
        (
            DictSource(jnp.ones(3)),
            DictSource(jnp.ones(3) * 2),
        )
    )
    lagged = combined.build_lagged_response(None)
    out = combined.evaluate_with_lagged_response(None, lagged)
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


def test_power_exchange_conservation_mask_and_derivatives():
    species = make_dummy_species()
    radius = jnp.linspace(0.0, 1.0, 10)
    density = jnp.asarray([2.5, 1.2, 1.1, 0.1])[:, None] * (1.0 - 0.2 * radius)
    pressure = density * (jnp.asarray([12.0, 8.0, 6.0, 5.0])[:, None] - radius)
    direction = (0.01 * jnp.cos(density), 0.02 * jnp.sin(pressure))
    cotangent = jnp.cos(pressure)
    for mode in ("all", "exclude_frozen"):
        for log_mode in ("neopax_simple", "ntssfusion"):
            source = PowerExchangeSource(
                species=species, mode=mode, coulomb_log_mode=log_mode,
                temperature_active_mask=jnp.asarray([True, True, True, False]),
            )

            def exchange(n, p):
                state = TransportState(density=n, pressure=p, Er=jnp.zeros(10))
                return source(state)["power_exchange"]

            value, tangent = jax.jvp(exchange, (density, pressure), direction)
            roundoff = 4 * np.finfo(value.dtype).eps
            np.testing.assert_allclose(value.sum(axis=0) / jnp.abs(value).sum(axis=0), 0.0, atol=roundoff)
            np.testing.assert_allclose(tangent.sum(axis=0) / jnp.abs(tangent).sum(axis=0), 0.0, atol=roundoff)
            if mode == "exclude_frozen":
                np.testing.assert_array_equal(value[-1], 0.0)
                np.testing.assert_array_equal(tangent[-1], 0.0)
            epsilon = 1e-5
            finite_difference = (
                exchange(density + epsilon * direction[0], pressure + epsilon * direction[1])
                - exchange(density - epsilon * direction[0], pressure - epsilon * direction[1])
            ) / (2 * epsilon)
            np.testing.assert_allclose(tangent, finite_difference, rtol=2e-7, atol=2e-9)
            _, pullback = jax.vjp(exchange, density, pressure)
            density_bar, pressure_bar = pullback(cotangent)
            np.testing.assert_allclose(
                jnp.vdot(cotangent, tangent),
                jnp.vdot(density_bar, direction[0]) + jnp.vdot(pressure_bar, direction[1]),
                rtol=2e-12, atol=2e-14,
            )


def test_power_exchange_fused_stage_repeats_are_identical():
    species = make_dummy_species()
    density = jnp.broadcast_to(jnp.asarray([2.5, 1.2, 1.1, 0.1])[:, None], (4, 51))
    pressure = density * (jnp.asarray([12.0, 8.0, 6.0, 5.0])[:, None] - jnp.linspace(0.0, 1.0, 51))
    source = PowerExchangeSource(species=species, mode="all")

    def pressure_rhs(p):
        state = TransportState(density=density, pressure=p, Er=jnp.zeros(51))
        return jnp.sin(p) + source(state)["power_exchange"]

    call = jax.jit(jax.vmap(pressure_rhs))
    stages = pressure[None, :, :] + jnp.arange(7)[:, None, None] * 1e-3
    expected = np.asarray(call(stages))
    for _ in range(30):
        np.testing.assert_array_equal(np.asarray(call(stages)), expected)


def test_power_exchange_temperature_equilibration():
    state = make_dummy_state()
    species = make_dummy_species()
    src = TemperatureEquilibrationPowerExchangeSource(mode="all")
    result = src(state, species)
    assert result["power_exchange_temperature_equilibration"].shape == state.temperature.shape


def test_power_exchange_temperature_equilibration_is_pairwise_conservative():
    density = jnp.asarray(
        [
            [0.35, 0.31],
            [0.35, 0.31],
        ]
    )
    temperature = jnp.asarray(
        [
            [6.7, 5.5],
            [1.0, 1.2],
        ]
    )
    state = TransportState(density=density, pressure=density * temperature, Er=jnp.zeros(2))
    species = Species(
        number_species=2,
        species_indices=jnp.arange(2),
        mass_mp=jnp.asarray([0.000544617, 1.0]),
        charge_qp=jnp.asarray([-1.0, 1.0]),
        names=("e", "ion"),
    )
    src = TemperatureEquilibrationPowerExchangeSource(mode="all")
    result = src(state, species)["power_exchange_temperature_equilibration"]
    assert jnp.allclose(jnp.sum(result, axis=0), jnp.zeros(result.shape[-1]), rtol=1e-10, atol=1e-10)


def test_bremsstrahlung_radiation():
    state = make_dummy_state()
    species = make_dummy_species()
    src = BremsstrahlungRadiationSource()
    result = src(state, species)
    assert result["PBrems"].shape == state.temperature[species.species_idx["e"]].shape
    assert result["Zeff"].shape == state.temperature[species.species_idx["e"]].shape


def make_two_temperature_state():
    # Species at a common temperature exchange no power, so the pairwise sum needs a spread.
    density = jnp.ones((4, 10))
    temperature = jnp.array([12.0, 8.0, 8.0, 8.0])[:, None] * jnp.ones((4, 10))
    return TransportState(density=density, pressure=density * temperature, Er=jnp.zeros(10))


def test_neopax_simple_coulomb_log_uses_transport_state_units():
    # The two modes are separate models and need not agree in general. At these conditions they
    # describe the same regime and agree to better than 1e-3, which makes the NTSSfusion branch a
    # usable reference for the simple one. A constant left in another unit system offsets
    # ln(Lambda) by roughly 16 and breaks the agreement at every state.
    state = make_two_temperature_state()
    species = make_dummy_species()
    simple = PowerExchangeSource(mode="all", coulomb_log_mode="neopax_simple")(state, species)
    ntss = PowerExchangeSource(mode="all", coulomb_log_mode="ntssfusion")(state, species)
    ratio = simple["power_exchange"] / ntss["power_exchange"]
    assert jnp.all(jnp.abs(ratio - 1.0) < 1.0e-3)


def test_bremsstrahlung_relativistic_correction_reads_temperature_as_kev():
    # The correction is a function of Te/(m_e c^2), so stating it in keV here is independent of the
    # units the implementation happens to work in. Reading the keV state as eV would shrink the
    # ratio by 1e3 and leave the correction indistinguishable from 1.
    state = make_two_temperature_state()
    species = make_dummy_species()
    plain, Zeff = bremsstrahlung_radiation_generalized(state, species=species)
    corrected, _ = bremsstrahlung_radiation_generalized(
        state, species=species, use_relativistic_correction=True
    )
    Te_keV = state.temperature[0]
    electron_rest_mass_keV = 511.0
    ratio = Te_keV / electron_rest_mass_keV
    expected = (1.0 + 2.0 * ratio) * (1.0 + (2.0 / Zeff) * (1.0 - 1.0 / (1.0 + ratio)))
    assert jnp.allclose(corrected / plain, expected, rtol=1e-10)
