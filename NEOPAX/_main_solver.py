"""Main transport solver orchestration for NEOPAX.

This module centralizes:
- Reading equation toggles from solver parameters.
- Assembly of equation classes from the registry.
- Flux/source coupling.
- Time integration via modular transport-solver backends.

It supports two electric field modes:
- diffusion: evolve Er using the Er transport equation.
- entropy: compute Er by ambipolar root selection and set dEr/dt = 0.
"""

from __future__ import annotations

from typing import Any
import dataclasses

import jax
import jax.numpy as jnp

from ._ambipolarity import find_ambipolar_Er_min_entropy_jit, find_ambipolar_Er_min_entropy_jit_multires
from ._neoclassical import get_Neoclassical_Fluxes
from ._state import TransportState
from ._transport_solvers import build_time_solver
from ._transport_flux_models import build_transport_flux_model
from ._transport_equations import get_equation


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


def _resize_toggle_array(arr: Any, n_species: int, default: bool) -> jax.Array:
    """Resize/pad generic toggle arrays to n_species."""
    if arr is None:
        return jnp.asarray([default] * n_species, dtype=bool)
    out = jnp.asarray(arr, dtype=bool)
    if out.ndim == 0:
        return jnp.asarray([bool(out)] * n_species, dtype=bool)
    if out.shape[0] >= n_species:
        return out[:n_species]
    pad = jnp.asarray([default] * (n_species - out.shape[0]), dtype=bool)
    return jnp.concatenate([out, pad], axis=0)


def _find_electron_index(charge_qp: Any) -> int:
    """Find electron species index as most negative charge state."""
    if charge_qp is None:
        return -1
    q = jnp.asarray(charge_qp)
    if q.ndim == 0 or q.shape[0] == 0:
        return -1
    eidx = int(jnp.argmin(q))
    return eidx if float(q[eidx]) < 0.0 else -1


def _as_transport_state(y: Any) -> TransportState:
    """Validate and return TransportState input."""
    if isinstance(y, TransportState):
        return y

    raise TypeError("State must be a TransportState instance.")


def _compute_flux_models(
    transport_flux_model: Any,
    initial_species: Any,
    state: TransportState,
    grid: Any,
    field: Any,
    database: Any,
    turbulent: Any,
    solver_parameters: Any,
    bc: dict[str, Any] | None = None,
) -> dict[str, jax.Array]:
    return transport_flux_model(
        initial_species,
        state,
        grid,
        field,
        database,
        turbulent,
        solver_parameters,
        bc,
    )


def _compute_entropy_mode_Er(
    initial_species: Any,
    state: TransportState,
    grid: Any,
    field: Any,
    database: Any,
    solver_parameters: Any,
    bc: dict[str, Any] | None = None,
) -> jax.Array:
    nrad = state.Er.shape[0]
    er_min = float(getattr(solver_parameters, "er_ambipolar_scan_min", -20.0))
    er_max = float(getattr(solver_parameters, "er_ambipolar_scan_max", 20.0))
    n_scan = int(getattr(solver_parameters, "er_ambipolar_n_scan", 96))
    tol = float(getattr(solver_parameters, "er_ambipolar_tol", 1e-6))
    maxiter = int(getattr(solver_parameters, "er_ambipolar_maxiter", 30))
    n_coarse = int(getattr(solver_parameters, "er_ambipolar_n_coarse", 24))
    n_fine = int(getattr(solver_parameters, "er_ambipolar_n_fine", 8))
    root_method = str(getattr(solver_parameters, "er_ambipolar_method", "jit_multires")).strip().lower()

    density_bc = bc.get("density") if isinstance(bc, dict) else None
    temperature_bc = bc.get("temperature") if isinstance(bc, dict) else None
    n_right, n_right_grad = _extract_right_constraints(density_bc, state.density)
    t_right, t_right_grad = _extract_right_constraints(temperature_bc, state.temperature)

    def best_root_for_radius(i: jax.Array) -> jax.Array:
        def gamma_of_er(er_val: jax.Array) -> jax.Array:
            er_vec = state.Er.at[i].set(er_val)
            _, gamma_neo, _, _ = get_Neoclassical_Fluxes(
                initial_species,
                grid,
                field,
                database,
                er_vec,
                state.temperature,
                state.density,
                density_right_constraint=n_right,
                density_right_grad_constraint=n_right_grad,
                temperature_right_constraint=t_right,
                temperature_right_grad_constraint=t_right_grad,
            )
            return jnp.sum(initial_species.charge_qp * gamma_neo[:, i])

        def entropy_of_er(er_val: jax.Array) -> jax.Array:
            er_vec = state.Er.at[i].set(er_val)
            _, gamma_neo, _, _ = get_Neoclassical_Fluxes(
                initial_species,
                grid,
                field,
                database,
                er_vec,
                state.temperature,
                state.density,
                density_right_constraint=n_right,
                density_right_grad_constraint=n_right_grad,
                temperature_right_constraint=t_right,
                temperature_right_grad_constraint=t_right_grad,
            )
            return jnp.sum(jnp.abs(gamma_neo[:, i]))

        if root_method == "jit_multires":
            best_er, _, _, _ = find_ambipolar_Er_min_entropy_jit_multires(
                gamma_of_er,
                entropy_of_er,
                Er_range=(er_min, er_max),
                n_coarse=n_coarse,
                n_fine=n_fine,
                tol=tol,
                x_tol=tol,
                maxiter=maxiter,
            )
        else:
            best_er, _, _, _ = find_ambipolar_Er_min_entropy_jit(
                gamma_of_er,
                entropy_of_er,
                Er_range=(er_min, er_max),
                n_scan=n_scan,
                tol=tol,
                x_tol=tol,
                maxiter=maxiter,
            )
        return best_er

    return jax.vmap(best_root_for_radius)(jnp.arange(nrad))


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class MainTransportModel:
    """Composed main transport model that couples selected equations."""

    solver_parameters: Any
    source_models: dict[str, Any] | None = None
    bc: dict[str, Any] | None = None
    transport_flux_model: Any | None = None

    def _rhs_transport_state(self, t: jax.Array, y: Any, args: tuple[Any, ...]) -> TransportState:
        del t
        initial_species, grid, field, database, turbulent, _ = args

        state = _as_transport_state(y)
        n_species = state.density.shape[0]
        evolve_density = _resize_toggle_array(
            getattr(self.solver_parameters, "evolve_density", None), n_species, default=True
        )
        evolve_temperature = _resize_toggle_array(
            getattr(self.solver_parameters, "evolve_temperature", None), n_species, default=True
        )
        evolve_er = bool(getattr(self.solver_parameters, "evolve_Er", True))
        density_rhs = jnp.zeros_like(state.density)
        temperature_rhs = jnp.zeros_like(state.temperature)
        er_rhs = jnp.zeros_like(state.Er)

        er_mode = getattr(self.solver_parameters, "er_mode", "diffusion")
        if er_mode == "entropy":
            er_for_physics = _compute_entropy_mode_Er(
                initial_species,
                state,
                grid,
                field,
                database,
                self.solver_parameters,
                self.bc,
            )
            state_for_physics = dataclasses.replace(state, Er=er_for_physics)
        else:
            state_for_physics = state

        flux_models = _compute_flux_models(
            self.transport_flux_model,
            initial_species,
            state_for_physics,
            grid,
            field,
            database,
            turbulent,
            self.solver_parameters,
            self.bc,
        )

        if jnp.any(evolve_density):
            density_eq = get_equation("density")()
            density_rhs = density_eq(
                state_for_physics,
                flux_models,
                self.source_models,
                field,
                bc=self.bc,
            )

        if jnp.any(evolve_temperature):
            temperature_eq = get_equation("temperature")()
            temperature_rhs = temperature_eq(
                state_for_physics,
                flux_models,
                self.source_models,
                field,
                bc=self.bc,
            )

        if evolve_er:
            if er_mode == "diffusion":
                er_eq = get_equation("Er")()
                er_rhs = er_eq(
                    state_for_physics,
                    flux_models,
                    self.source_models,
                    field,
                    self.solver_parameters,
                    bc=self.bc,
                    charge_qp=initial_species.charge_qp,
                    species_mass=initial_species.mass,
                )
            else:
                er_rhs = jnp.zeros_like(state.Er)

        # Apply per-species boolean toggles: zero out RHS for frozen species.
        density_rhs = density_rhs * evolve_density[:, None]
        temperature_rhs = temperature_rhs * evolve_temperature[:, None]

        # Quasi-neutrality: electron density is not independent — it is derived from
        # n_e = sum_ions Z_i * n_i  =>  dn_e/dt = sum_ions Z_i * dn_i/dt
        # Build an ion-only mask (all species except the electron) so the sum is
        # explicit and independent of whether the electron row is zero or not.
        eidx = _find_electron_index(getattr(initial_species, "charge_qp", None))
        if eidx >= 0:
            n_sp = density_rhs.shape[0]
            ion_mask = jnp.arange(n_sp) != eidx  # shape (n_species,), True for ions
            qn_rhs = jnp.sum(
                initial_species.charge_qp[:, None] * density_rhs * ion_mask[:, None], axis=0
            )
            density_rhs = density_rhs.at[eidx, :].set(qn_rhs)

        return TransportState(
            density=density_rhs,
            temperature=temperature_rhs,
            Er=er_rhs,
            species_names=state.species_names,
            is_evolved=state.is_evolved,
        )

    def vector_field(self, t: jax.Array, y: Any, args: tuple[Any, ...]) -> Any:
        rhs_state = self._rhs_transport_state(t, y, args)
        return rhs_state


def main_transport_solver(
    state: Any,
    args: tuple[Any, ...],
    source_models: dict[str, Any] | None = None,
    bc: dict[str, Any] | None = None,
) -> Any:
    """Evaluate one RHS call of the composed transport system."""
    solver_parameters = args[-1]
    transport_flux_model = build_transport_flux_model(solver_parameters)
    model = MainTransportModel(
        solver_parameters,
        source_models=source_models,
        bc=bc,
        transport_flux_model=transport_flux_model,
    )
    return model.vector_field(jnp.asarray(0.0), state, args)


def solve_transport_equations(
    state0: Any,
    args: tuple[Any, ...],
    source_models: dict[str, Any] | None = None,
    bc: dict[str, Any] | None = None,
    solver: Any = None,
):
    """Integrate the selected transport equations in time via TransportSolver backend."""
    solver_parameters = args[-1]

    transport_flux_model = build_transport_flux_model(solver_parameters)
    model = MainTransportModel(
        solver_parameters,
        source_models=source_models,
        bc=bc,
        transport_flux_model=transport_flux_model,
    )
    backend = build_time_solver(solver_parameters, solver_override=solver)
    return backend.solve(state0, model.vector_field, args=args)
