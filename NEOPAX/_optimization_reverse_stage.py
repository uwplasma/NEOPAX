"""Static reverse-AD stage selection for optimization.

This module deliberately contains no transport equations and no JAX tracing.
It owns the *lifetime* and model compatibility checks for a future persistent
reverse stage.  The default benchmark path remains unstaged until an adapter
provides an explicit dynamic-input kernel.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax


_EXACT_LIJ_MODEL = "ntx_exact_lij_runtime"


def _find_exact_lij_model(model: Any) -> Any | None:
    """Return the nested exact-Lij model without importing reverse helpers.

    The structural capability check avoids coupling stage selection to a
    particular composite transport-model wrapper.
    """

    if (
        model is not None
        and callable(getattr(model, "with_support_payload", None))
        and callable(getattr(model, "_solve_lij_prepared_local", None))
    ):
        return model
    if dataclasses.is_dataclass(model) and not isinstance(model, type):
        for field in dataclasses.fields(model):
            found = _find_exact_lij_model(getattr(model, field.name))
            if found is not None:
                return found
    return None


@dataclasses.dataclass(frozen=True, slots=True)
class ExactLijReverseStageAdapter:
    """Static exact-Lij adapter selected once for an optimization stage.

    This is intentionally only a selector/validator for now.  The dynamic
    residual and pullback kernels will be added here only after they accept
    state, Er, and support explicitly instead of constructing state-capturing
    evaluator closures.
    """

    model_name: str
    radial_count: int
    species_count: int
    local_particle_flux_kernel: Any

    def local_particle_flux_evaluator(self, state: Any, *, geometry: Any, support: Any):
        """Return the small staged local-flux call used by root operations.

        The wrapper is intentionally trivial: its captured values are passed
        immediately as dynamic arguments to the persistent kernel.  The exact
        Lij model object captured by that kernel is the fixed stage model.
        """

        def evaluator(radius_index, er_value):
            return self.local_particle_flux_kernel(
                state,
                geometry,
                support,
                radius_index,
                er_value,
            )

        return evaluator

    def validate_runtime(self, runtime: Any) -> None:
        model = _find_exact_lij_model(runtime.models.flux)
        if model is None:
            raise ValueError(
                "The exact-Lij optimization reverse stage requires an "
                "NTX exact-Lij transport model in runtime.models.flux."
            )
        if int(model.species.number_species) != self.species_count:
            raise ValueError("Exact-Lij optimization reverse-stage species layout changed after construction.")


@dataclasses.dataclass(frozen=True, slots=True)
class GeometryTransportReverseStage:
    """One optimization-lifetime reverse-stage selection.

    The object contains static model/layout information only.  Numerical
    quantities such as optimizer DoFs, state, Er, support values, and Radau
    masks are not stored here and will be dynamic inputs to later kernels.
    """

    model_name: str
    adapter: ExactLijReverseStageAdapter

    def validate_runtime(self, runtime: Any) -> None:
        self.adapter.validate_runtime(runtime)


def geometry_transport_reverse_stage(*, config: dict, runtime: Any, radial_count: int) -> GeometryTransportReverseStage:
    """Build the selected optimization reverse stage before a run begins.

    Unsupported TOML model selections fail explicitly.  In particular, this
    must not fall back from an unimplemented database/runtime-scan AD adapter
    to the exact-Lij path.
    """

    model_name = str(config.get("neoclassical", {}).get("flux_model", "ntx_database")).strip().lower()
    if model_name != _EXACT_LIJ_MODEL:
        raise NotImplementedError(
            "Optimization reverse staging is currently implemented only for "
            f"'{_EXACT_LIJ_MODEL}'; TOML selected {model_name!r}."
        )
    exact_model = _find_exact_lij_model(runtime.models.flux)
    if exact_model is None:
        raise ValueError(
            "TOML selected ntx_exact_lij_runtime, but the runtime does not contain "
            "an exact-Lij model with explicit support-payload capability."
        )
    if not callable(getattr(exact_model, "local_particle_flux_from_state_support", None)):
        raise ValueError(
            "The exact-Lij runtime lacks the explicit local-flux boundary required "
            "for optimization reverse staging."
        )

    def _local_particle_flux_kernel(state, geometry, support, radius_index, er_value):
        return exact_model.local_particle_flux_from_state_support(
            state,
            radius_index,
            er_value,
            geometry=geometry,
            support=support,
        )

    adapter = ExactLijReverseStageAdapter(
        model_name=model_name,
        radial_count=int(radial_count),
        species_count=int(exact_model.species.number_species),
        local_particle_flux_kernel=jax.jit(_local_particle_flux_kernel),
    )
    stage = GeometryTransportReverseStage(model_name=model_name, adapter=adapter)
    stage.validate_runtime(runtime)
    return stage
