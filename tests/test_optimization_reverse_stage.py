"""Unit checks for exact-Lij optimization reverse-stage selection."""

import dataclasses
from types import SimpleNamespace

import pytest

from NEOPAX._optimization_reverse_stage import geometry_transport_reverse_stage


@dataclasses.dataclass(frozen=True)
class ExactModel:
    species: object

    def with_support_payload(self, support):
        del support
        return self

    def _solve_lij_prepared_local(self):
        return None

    def local_particle_flux_from_state_support(self, *args, **kwargs):
        del args, kwargs
        return None


@dataclasses.dataclass(frozen=True)
class CompositeModel:
    neoclassical_model: object


def _runtime(model, species_count=3):
    return SimpleNamespace(
        models=SimpleNamespace(flux=model),
        species=SimpleNamespace(number_species=species_count),
    )


def test_exact_lij_stage_selects_nested_exact_model():
    runtime = _runtime(CompositeModel(ExactModel(SimpleNamespace(number_species=3))))

    stage = geometry_transport_reverse_stage(
        config={"neoclassical": {"flux_model": "ntx_exact_lij_runtime"}},
        runtime=runtime,
        radial_count=51,
    )

    assert stage.model_name == "ntx_exact_lij_runtime"
    assert stage.adapter.radial_count == 51
    stage.validate_runtime(runtime)


def test_stage_rejects_unimplemented_toml_model_without_fallback():
    runtime = _runtime(CompositeModel(ExactModel(SimpleNamespace(number_species=3))))

    with pytest.raises(NotImplementedError, match="ntx_exact_lij_runtime"):
        geometry_transport_reverse_stage(
            config={"neoclassical": {"flux_model": "ntx_database"}},
            runtime=runtime,
            radial_count=51,
        )


def test_stage_rejects_exact_lij_toml_when_runtime_lacks_exact_model():
    runtime = _runtime(SimpleNamespace())

    with pytest.raises(ValueError, match="does not contain"):
        geometry_transport_reverse_stage(
            config={"neoclassical": {"flux_model": "ntx_exact_lij_runtime"}},
            runtime=runtime,
            radial_count=51,
        )
