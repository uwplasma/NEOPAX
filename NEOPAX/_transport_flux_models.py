from __future__ import annotations

from typing import Any, Callable

import dataclasses
import jax
import jax.numpy as jnp

from ._neoclassical import (
    get_Neoclassical_Fluxes,
    get_Neoclassical_Fluxes_With_Momentum_Correction,
)
from ._species import (
    get_Thermodynamical_Forces_A1,
    get_Thermodynamical_Forces_A2,
    get_Thermodynamical_Forces_A3,
)
from ._cell_variable import get_gradient_density, get_gradient_temperature
from ._state import get_v_thermal
from ._turbulence import get_Turbulent_Fluxes_Analytical


TRANSPORT_FLUX_MODEL_REGISTRY: dict[str, Callable[[], "TransportFluxModelBase"]] = {}


def register_transport_flux_model(name: str, builder: Callable[[], "TransportFluxModelBase"]) -> None:
    TRANSPORT_FLUX_MODEL_REGISTRY[str(name).strip().lower()] = builder


def get_transport_flux_model(name: str) -> "TransportFluxModelBase":
    key = str(name).strip().lower()
    if key not in TRANSPORT_FLUX_MODEL_REGISTRY:
        raise ValueError(f"Unknown transport flux model '{name}'.")
    return TRANSPORT_FLUX_MODEL_REGISTRY[key]()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class TransportFluxModelBase:
    """Base class for any transport flux model.

    Implementations return a dict with optional keys such as:
      Gamma, Q
    """

    def __call__(
        self,
        species: Any,
        state: Any,
        grid: Any,
        field: Any,
        database: Any,
        turbulence: Any,
        solver_parameters: Any,
        bc: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError


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


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class CombinedTransportFluxModel(TransportFluxModelBase):
    models: tuple[TransportFluxModelBase, ...] = ()

    def __call__(
        self,
        species: Any,
        state: Any,
        grid: Any,
        field: Any,
        database: Any,
        turbulence: Any,
        solver_parameters: Any,
        bc: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        gamma_total = None
        q_total = None
        upar_total = None
        out: dict[str, Any] = {}

        for model in self.models:
            fluxes = model(species, state, grid, field, database, turbulence, solver_parameters, bc)
            gamma = fluxes.get("Gamma")
            q = fluxes.get("Q")
            upar = fluxes.get("Upar")

            if gamma is not None:
                gamma_total = gamma if gamma_total is None else gamma_total + gamma
            if q is not None:
                q_total = q if q_total is None else q_total + q
            if upar is not None:
                upar_total = upar if upar_total is None else upar_total + upar

            out.update(fluxes)

        if gamma_total is None:
            gamma_total = jnp.zeros_like(state.density)
        if q_total is None:
            q_total = jnp.zeros_like(state.density)
        if upar_total is None:
            upar_total = jnp.zeros_like(state.density)

        out["Gamma_total"] = gamma_total
        out["Q_total"] = q_total
        out["Upar_total"] = upar_total
        return out


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class MonkesNeoclassicalTransportModel(TransportFluxModelBase):
    def __call__(
        self,
        species: Any,
        state: Any,
        grid: Any,
        field: Any,
        database: Any,
        turbulence: Any,
        solver_parameters: Any,
        bc: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del turbulence, solver_parameters
        density_bc = bc.get("density") if isinstance(bc, dict) else None
        temperature_bc = bc.get("temperature") if isinstance(bc, dict) else None
        n_right, n_right_grad = _extract_right_constraints(density_bc, state.density)
        t_right, t_right_grad = _extract_right_constraints(temperature_bc, state.temperature)
        _, gamma_neo, q_neo, upar_neo = get_Neoclassical_Fluxes(
            species,
            grid,
            field,
            database,
            state.Er,
            state.temperature,
            state.density,
            density_right_constraint=n_right,
            density_right_grad_constraint=n_right_grad,
            temperature_right_constraint=t_right,
            temperature_right_grad_constraint=t_right_grad,
        )
        return {
            "Gamma": gamma_neo,
            "Q": q_neo,
            "Upar": upar_neo,
            "Gamma_neo": gamma_neo,
            "Q_neo": q_neo,
            "Upar_neo": upar_neo,
        }


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class MonkesMomentumTransportModel(TransportFluxModelBase):
    def __call__(
        self,
        species: Any,
        state: Any,
        grid: Any,
        field: Any,
        database: Any,
        turbulence: Any,
        solver_parameters: Any,
        bc: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del turbulence, solver_parameters

        # Build the same thermodynamic-force inputs as the standard neoclassical model,
        # then evaluate the momentum-corrected transport kernel.
        v_thermal = get_v_thermal(species.mass, state.temperature)
        r_grid = field.r_grid
        r_grid_half = field.r_grid_half
        dr = field.dr

        density_bc = bc.get("density") if isinstance(bc, dict) else None
        temperature_bc = bc.get("temperature") if isinstance(bc, dict) else None
        n_right, n_right_grad = _extract_right_constraints(density_bc, state.density)
        t_right, t_right_grad = _extract_right_constraints(temperature_bc, state.temperature)

        dndr = jax.vmap(
            lambda n_a, n_rc, n_rg: get_gradient_density(
                n_a,
                r_grid,
                r_grid_half,
                dr,
                right_face_constraint=n_rc,
                right_face_grad_constraint=n_rg,
            )
        )(state.density, n_right, n_right_grad)

        dTdr = jax.vmap(
            lambda t_a, t_rc, t_rg: get_gradient_temperature(
                t_a,
                r_grid,
                r_grid_half,
                dr,
                right_face_constraint=t_rc,
                right_face_grad_constraint=t_rg,
            )
        )(state.temperature, t_right, t_right_grad)

        a1 = jax.vmap(
            lambda z_a, n_a, t_a, dn_a, dT_a: get_Thermodynamical_Forces_A1(
                z_a,
                n_a,
                t_a,
                dn_a,
                dT_a,
                state.Er,
            )
        )(species.charge, state.density, state.temperature, dndr, dTdr)
        a2 = jax.vmap(get_Thermodynamical_Forces_A2)(state.temperature, dTdr)
        a3 = get_Thermodynamical_Forces_A3(state.Er)

        correction = get_Neoclassical_Fluxes_With_Momentum_Correction(
            grid,
            field,
            database,
            species.mass,
            species.charge,
            state.Er,
            state.temperature,
            state.density,
            v_thermal,
            a1,
            a2,
            a3,
            dndr,
            dTdr,
        )

        def _species_radial(x: Any) -> Any:
            arr = jnp.asarray(x)
            target_shape = state.density.shape
            if arr.ndim == 2 and arr.shape == (target_shape[1], target_shape[0]):
                return jnp.swapaxes(arr, 0, 1)
            return arr

        if isinstance(correction, tuple) and len(correction) >= 2:
            gamma_neo = _species_radial(correction[0])
            q_neo = _species_radial(correction[1])
            out = {
                "Gamma": gamma_neo,
                "Q": q_neo,
                "Gamma_neo": gamma_neo,
                "Q_neo": q_neo,
            }
            if len(correction) >= 3:
                upar_neo = _species_radial(correction[2])
                out["Upar"] = upar_neo
                out["Upar_neo"] = upar_neo
            return out
        return MonkesNeoclassicalTransportModel()(
            species,
            state,
            grid,
            field,
            database,
            turbulence,
            solver_parameters,
            bc,
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class ZeroTurbulentTransportModel(TransportFluxModelBase):
    def __call__(
        self,
        species: Any,
        state: Any,
        grid: Any,
        field: Any,
        database: Any,
        turbulence: Any,
        solver_parameters: Any,
        bc: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del species, grid, field, database, turbulence, solver_parameters, bc
        gamma_turb = jnp.zeros_like(state.density)
        q_turb = jnp.zeros_like(state.density)
        return {
            "Gamma": gamma_turb,
            "Q": q_turb,
            "Gamma_turb": gamma_turb,
            "Q_turb": q_turb,
        }


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class TurbulenceStateTransportModel(TransportFluxModelBase):
    def __call__(
        self,
        species: Any,
        state: Any,
        grid: Any,
        field: Any,
        database: Any,
        turbulence: Any,
        solver_parameters: Any,
        bc: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del grid, field, database, bc
        if turbulence is None:
            return ZeroTurbulentTransportModel()(
                species,
                state,
                grid,
                field,
                database,
                turbulence,
                solver_parameters,
            )

        q_turb = getattr(turbulence, "Q_turb", None)
        if q_turb is None:
            q_turb = getattr(turbulence, "Qa_turb", None)
        if q_turb is None:
            q_turb = jnp.zeros_like(state.density)

        gamma_turb = getattr(turbulence, "Gamma_turb", None)
        if gamma_turb is None:
            gamma_turb = jnp.zeros_like(state.density)

        out = {
            "Gamma": gamma_turb,
            "Q": q_turb,
            "Gamma_turb": gamma_turb,
            "Q_turb": q_turb,
        }

        upar_turb = getattr(turbulence, "Upar_turb", None)
        if upar_turb is not None:
            out["Upar"] = upar_turb
            out["Upar_turb"] = upar_turb
        return out


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class AnalyticalTurbulentTransportModel(TransportFluxModelBase):
    def __call__(
        self,
        species: Any,
        state: Any,
        grid: Any,
        field: Any,
        database: Any,
        turbulence: Any,
        solver_parameters: Any,
        bc: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del database, turbulence
        n_species = state.density.shape[0]

        chi_t = jnp.asarray(getattr(solver_parameters, "chi_temperature", jnp.ones(n_species) * 0.5))
        chi_n = jnp.asarray(getattr(solver_parameters, "chi_density", jnp.zeros(n_species)))

        if chi_t.shape[0] < n_species:
            chi_t = jnp.pad(chi_t, (0, n_species - chi_t.shape[0]), constant_values=0.5)
        else:
            chi_t = chi_t[:n_species]

        if chi_n.shape[0] < n_species:
            chi_n = jnp.pad(chi_n, (0, n_species - chi_n.shape[0]), constant_values=0.0)
        else:
            chi_n = chi_n[:n_species]

        density_bc = bc.get("density") if isinstance(bc, dict) else None
        temperature_bc = bc.get("temperature") if isinstance(bc, dict) else None
        n_right, n_right_grad = _extract_right_constraints(density_bc, state.density)
        t_right, t_right_grad = _extract_right_constraints(temperature_bc, state.temperature)

        gamma_turb, q_turb = get_Turbulent_Fluxes_Analytical(
            species,
            grid,
            chi_t,
            chi_n,
            state.temperature,
            state.density,
            field,
            density_right_constraint=n_right,
            density_right_grad_constraint=n_right_grad,
            temperature_right_constraint=t_right,
            temperature_right_grad_constraint=t_right_grad,
        )

        return {
            "Gamma": gamma_turb,
            "Q": q_turb,
            "Gamma_turb": gamma_turb,
            "Q_turb": q_turb,
        }


def build_transport_flux_model(solver_parameters: Any) -> CombinedTransportFluxModel:
    """Build the active composed transport model from runtime parameters."""
    neoclassical_name = str(
        getattr(solver_parameters, "neoclassical_transport_model", "neoclassical")
    ).strip().lower()
    turbulent_name = str(
        getattr(solver_parameters, "turbulent_transport_model", "from_turbulence_state")
    ).strip().lower()

    return CombinedTransportFluxModel(
        models=(
            get_transport_flux_model(neoclassical_name),
            get_transport_flux_model(turbulent_name),
        )
    )

register_transport_flux_model("monkes_database", MonkesNeoclassicalTransportModel)
register_transport_flux_model("neoclassical", MonkesNeoclassicalTransportModel)
register_transport_flux_model("neoclassical_momentum", MonkesMomentumTransportModel)
register_transport_flux_model("none", ZeroTurbulentTransportModel)
register_transport_flux_model("zero", ZeroTurbulentTransportModel)
register_transport_flux_model("from_turbulence_state", TurbulenceStateTransportModel)
register_transport_flux_model("analytical", AnalyticalTurbulentTransportModel)
