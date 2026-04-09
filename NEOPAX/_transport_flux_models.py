from __future__ import annotations


from typing import Any, Callable
import abc
import dataclasses
import jax
import jax.numpy as jnp
from ._neoclassical import get_Neoclassical_Fluxes,get_Neoclassical_Fluxes_With_Momentum_Correction
from ._turbulence import get_Turbulent_Fluxes_Analytical



# Registry for modular selection
TRANSPORT_FLUX_MODEL_REGISTRY: dict[str, Callable[[], "TransportFluxModelBase"]] = {}

def register_transport_flux_model(name: str, builder: Callable[[], "TransportFluxModelBase"]) -> None:
    TRANSPORT_FLUX_MODEL_REGISTRY[str(name).strip().lower()] = builder

def get_transport_flux_model(name: str) -> "TransportFluxModelBase":
    key = str(name).strip().lower()
    if key not in TRANSPORT_FLUX_MODEL_REGISTRY:
        raise ValueError(f"Unknown transport flux model '{name}'.")
    return TRANSPORT_FLUX_MODEL_REGISTRY[key]()



@dataclasses.dataclass(frozen=True, eq=False)
class TransportFluxModelBase(abc.ABC):
        """
        Abstract base class for transport flux models.
        Output dict keys:
            - Gamma: particle flux
            - Q: heat flux
            - Upar: parallel flow
        """
        @abc.abstractmethod
        def __call__(self, state, geometry=None, params=None) -> dict:
                pass


def _extract_right_constraints(bc_model: Any, state_arr: jax.Array) -> tuple[jax.Array, jax.Array]:
    n_species = state_arr.shape[0]
    default_value = state_arr[:, -1]
    default_grad = jnp.zeros_like(default_value)
    if bc_model is None:
        return default_value, default_grad

    right_type = str(getattr(bc_model, "right_type", "dirichlet")).strip().lower()

    def _as_species(arr, fallback):
        if arr is None:
            return fallback
        out = jnp.asarray(arr)
        if out.ndim == 0:
            out = jnp.repeat(out[None], n_species, axis=0)
        if out.shape[0] < n_species:
            out = jnp.pad(out, (0, n_species - out.shape[0]), mode="edge")
        return out[:n_species]

    right_value = _as_species(getattr(bc_model, "right_value", None), default_value)
    right_grad = _as_species(getattr(bc_model, "right_gradient", None), default_grad)
    right_decay = _as_species(getattr(bc_model, "right_decay_length", None), jnp.ones_like(default_value))

    if right_type == "dirichlet":
        return right_value, jnp.zeros_like(right_value)
    if right_type == "neumann":
        return default_value, right_grad
    if right_type == "robin":
        robin_grad = -right_value / (right_decay + 1e-12)
        return default_value, robin_grad
    return default_value, default_grad



@dataclasses.dataclass(frozen=True, eq=False)
class CombinedTransportFluxModel(TransportFluxModelBase):
    neoclassical_model: TransportFluxModelBase
    turbulent_model: TransportFluxModelBase

    def __call__(self, state, geometry=None, params=None) -> dict:
        # Call both models and sum outputs
        neo = self.neoclassical_model(state, geometry, params)
        turb = self.turbulent_model(state, geometry, params)
        out = {}
        for key in ("Gamma", "Q", "Upar"):
            out[key] = neo.get(key, 0) + turb.get(key, 0)
        out["Gamma_neo"] = neo.get("Gamma", 0)
        out["Q_neo"] = neo.get("Q", 0)
        out["Upar_neo"] = neo.get("Upar", 0)
        out["Gamma_turb"] = turb.get("Gamma", 0)
        out["Q_turb"] = turb.get("Q", 0)
        out["Upar_turb"] = turb.get("Upar", 0)
        return out



@dataclasses.dataclass(frozen=True, eq=False)
class MonkesDatabaseTransportModel(TransportFluxModelBase):
    def __call__(self, state, geometry=None, params=None) -> dict:
        # Assume params contains all needed info (species, grid, etc.)
        # This is a minimal, jittable, differentiable wrapper
        _, gamma_neo, q_neo, upar_neo = get_Neoclassical_Fluxes(
            params["species"],
            params["grid"],
            params["field"],
            params["neoclassical_data"],
            state.Er,
            state.temperature,
            state.density,
        )
        return {
            "Gamma": gamma_neo,
            "Q": q_neo,
            "Upar": upar_neo,
        }


@dataclasses.dataclass(frozen=True, eq=False)
class MonkesDatabaseWithMomentumTransportModel(TransportFluxModelBase):
    def __call__(self, state, geometry=None, params=None) -> dict:
        # Use new unified get_Neoclassical_Fluxes_With_Momentum_Correction interface
        # Pass boundary constraints if present in params
        density_right_constraint = params.get("density_right_constraint", None)
        density_right_grad_constraint = params.get("density_right_grad_constraint", None)
        temperature_right_constraint = params.get("temperature_right_constraint", None)
        temperature_right_grad_constraint = params.get("temperature_right_grad_constraint", None)
        correction = get_Neoclassical_Fluxes_With_Momentum_Correction(
            params["species"],
            params["grid"],
            params["field"],
            params["neoclassical_data"],
            state.Er,
            state.temperature,
            state.density,
            density_right_constraint=density_right_constraint,
            density_right_grad_constraint=density_right_grad_constraint,
            temperature_right_constraint=temperature_right_constraint,
            temperature_right_grad_constraint=temperature_right_grad_constraint,
        )
        Gamma, Q, Upar, *_ = correction
        return {"Gamma": Gamma, "Q": Q, "Upar": Upar}


@dataclasses.dataclass(frozen=True, eq=False)
class ZeroTransportModel(TransportFluxModelBase):
    def __call__(self, state, geometry=None, params=None) -> dict:
        gamma = jnp.zeros_like(state.density)
        q = jnp.zeros_like(state.density)
        upar = jnp.zeros_like(state.density)
        return {"Gamma": gamma, "Q": q, "Upar": upar}





@dataclasses.dataclass(frozen=True, eq=False)
class AnalyticalTurbulentTransportModel(TransportFluxModelBase):
    def __call__(self, state, geometry=None, params=None) -> dict:
        # Assume params contains all needed info (chi_t, chi_n, etc.)
        gamma_turb, q_turb = get_Turbulent_Fluxes_Analytical(
            params["species"],
            params["grid"],
            params["chi_t"],
            params["chi_n"],
            state.temperature,
            state.density,
            params["field"],
        )
        upar = jnp.zeros_like(state.density)
        return {"Gamma": gamma_turb, "Q": q_turb, "Upar": upar}



def build_transport_flux_model(params: Any) -> CombinedTransportFluxModel:
    """
    Build the active composed transport model from runtime parameters.
    Expects params["neoclassical_flux_model"] and params["turbulent_flux_model"] to specify model names.
    """
    neo_model = get_transport_flux_model(params["neoclassical_flux_model"])
    turb_model = get_transport_flux_model(params["turbulent_flux_model"])
    return CombinedTransportFluxModel(neo_model, turb_model)

register_transport_flux_model("monkes_database", MonkesDatabaseTransportModel)
register_transport_flux_model("monkes_database_with_momentum", MonkesDatabaseWithMomentumTransportModel)
register_transport_flux_model("turbulent_analytical", AnalyticalTurbulentTransportModel)
register_transport_flux_model("none", ZeroTransportModel)
