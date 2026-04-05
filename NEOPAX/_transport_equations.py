
from typing import Dict, Type
import dataclasses
import jax
import jax.numpy as jnp
import interpax
from jax import jit
from ._fem import conservative_update, faces_from_cell_centered
from ._cell_variable import make_profile_cell_variable
from ._boundary_conditions import right_constraints_from_bc_model
from ._constants import elementary_charge


@jit
def _plasma_permitivity(state, species_mass, field, grid_x):
    """Local plasma permittivity at grid_x for the Er diffusion equation."""
    psi_fac = 1.0 + 1.0 / (field.enlogation * jnp.square(field.iota))
    psi_fac = psi_fac.at[0].set(1.0)
    mass_density = jnp.sum(species_mass[:, None] * state.density, axis=0)
    epsilon_r = mass_density * psi_fac / jnp.square(field.B0)
    return interpax.Interpolator1D(field.r_grid, epsilon_r, extrap=True)(grid_x)

# --- Modular Equation Registry and Base ---
__equation_registry: Dict[str, Type] = {}

def register_equation(name: str):
    """Decorator to register equation classes in the registry."""
    def decorator(cls):
        __equation_registry[name] = cls
        return cls
    return decorator

@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class EquationBase:
    """
    Base class for transport equations. Subclasses must implement __call__.
    Base class for transport equations.
    """
    name: str = "base"

    def __call__(self, state, flux_models, source_models, field, **kwargs):
        raise NotImplementedError

def get_equation(name: str) -> Type:
    return __equation_registry[name]

def list_equations():
    return list(__equation_registry.keys())

# --- Example built-in equation: Density evolution ---
@register_equation("density")
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class DensityEquation(EquationBase):
    name: str = "density"
    reconstruction: str = "linear"

    def __call__(self, state, flux_models, source_models, field, bc=None, **kwargs):
        dr_cells = jnp.diff(field.r_grid_half)
        Vprime = field.Vprime
        Vprime_half = field.Vprime_half
        Gamma_total = flux_models.get("Gamma_total")
        bc_density = bc.get("density") if bc is not None else None

        if bc_density is not None and hasattr(bc_density, "right_type"):
            rv, rg = right_constraints_from_bc_model(bc_density, Gamma_total[:, -1])
            if rv is not None:
                Gamma_faces = jax.vmap(
                    lambda G, right_val: make_profile_cell_variable(
                        G,
                        field.r_grid_half,
                        left_face_constraint=jnp.asarray(0.0, dtype=G.dtype),
                        right_face_constraint=right_val,
                    ).face_value(reconstruction=self.reconstruction)
                )(Gamma_total, jnp.asarray(rv))
            else:
                Gamma_faces = jax.vmap(
                    lambda G, right_grad: make_profile_cell_variable(
                        G,
                        field.r_grid_half,
                        left_face_constraint=jnp.asarray(0.0, dtype=G.dtype),
                        right_face_grad_constraint=right_grad,
                    ).face_value(reconstruction=self.reconstruction)
                )(Gamma_total, jnp.asarray(rg))
        # Backward-compatible path for legacy BC objects.
        elif bc_density is not None and hasattr(bc_density, "apply_ghost"):
            if hasattr(bc_density, "apply_ghost_all"):
                Gamma_ghost = bc_density.apply_ghost_all(Gamma_total)
            else:
                Gamma_ghost = jax.vmap(lambda G: bc_density.apply_ghost(G))(Gamma_total)
            Gamma_faces = jax.vmap(faces_from_cell_centered)(Gamma_ghost)
        else:
            Gamma_faces = jax.vmap(
                lambda G: make_profile_cell_variable(
                    G,
                    field.r_grid_half,
                    left_face_constraint=jnp.asarray(0.0, dtype=G.dtype),
                    right_face_grad_constraint=jnp.asarray(0.0, dtype=G.dtype),
                ).face_value(reconstruction=self.reconstruction)
            )(Gamma_total)

        density_rhs = jax.vmap(
            lambda flux: conservative_update(flux, dr_cells, Vprime, Vprime_half)
        )(Gamma_faces)
        if source_models is not None and "density" in source_models:
            density_rhs += source_models["density"](state)
        return density_rhs

# --- Example built-in equation: Temperature evolution ---
@register_equation("temperature")
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class TemperatureEquation(EquationBase):
    name: str = "temperature"
    reconstruction: str = "linear"

    def __call__(self, state, flux_models, source_models, field, bc=None, **kwargs):
        dr_cells = jnp.diff(field.r_grid_half)
        Vprime = field.Vprime
        Vprime_half = field.Vprime_half
        Q_total = flux_models.get("Q_total")
        bc_temperature = bc.get("temperature") if bc is not None else None

        if bc_temperature is not None and hasattr(bc_temperature, "right_type"):
            rv, rg = right_constraints_from_bc_model(bc_temperature, Q_total[:, -1])
            if rv is not None:
                Q_faces = jax.vmap(
                    lambda Q, right_val: make_profile_cell_variable(
                        Q,
                        field.r_grid_half,
                        left_face_constraint=jnp.asarray(0.0, dtype=Q.dtype),
                        right_face_constraint=right_val,
                    ).face_value(reconstruction=self.reconstruction)
                )(Q_total, jnp.asarray(rv))
            else:
                Q_faces = jax.vmap(
                    lambda Q, right_grad: make_profile_cell_variable(
                        Q,
                        field.r_grid_half,
                        left_face_constraint=jnp.asarray(0.0, dtype=Q.dtype),
                        right_face_grad_constraint=right_grad,
                    ).face_value(reconstruction=self.reconstruction)
                )(Q_total, jnp.asarray(rg))
        # Backward-compatible path for legacy BC objects.
        elif bc_temperature is not None and hasattr(bc_temperature, "apply_ghost"):
            if hasattr(bc_temperature, "apply_ghost_all"):
                Q_ghost = bc_temperature.apply_ghost_all(Q_total)
            else:
                Q_ghost = jax.vmap(lambda Q: bc_temperature.apply_ghost(Q))(Q_total)
            Q_faces = jax.vmap(faces_from_cell_centered)(Q_ghost)
        else:
            Q_faces = jax.vmap(
                lambda Q: make_profile_cell_variable(
                    Q,
                    field.r_grid_half,
                    left_face_constraint=jnp.asarray(0.0, dtype=Q.dtype),
                    right_face_grad_constraint=jnp.asarray(0.0, dtype=Q.dtype),
                ).face_value(reconstruction=self.reconstruction)
            )(Q_total)

        temp_rhs = jax.vmap(
            lambda flux: conservative_update(flux, dr_cells, Vprime, Vprime_half)
        )(Q_faces)
        if source_models is not None and "temperature" in source_models:
            temp_rhs += source_models["temperature"](state)
        return temp_rhs

# --- Example built-in equation: Electric field (Er) evolution ---
@register_equation("Er")
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class ElectricFieldEquation(EquationBase):
    name: str = "Er"
    reconstruction: str = "linear"

    def _build_er_faces(self, er_profile, field, bc):
        bc_er = bc.get("Er") if bc is not None else None
        if bc_er is not None and hasattr(bc_er, "right_type"):
            rv_er, rg_er = right_constraints_from_bc_model(bc_er, er_profile[-1])
            if rv_er is not None:
                return make_profile_cell_variable(
                    er_profile,
                    field.r_grid_half,
                    left_face_grad_constraint=jnp.asarray(0.0, dtype=er_profile.dtype),
                    right_face_constraint=jnp.asarray(rv_er).reshape(-1)[0],
                ).face_value(reconstruction=self.reconstruction)

            return make_profile_cell_variable(
                er_profile,
                field.r_grid_half,
                left_face_grad_constraint=jnp.asarray(0.0, dtype=er_profile.dtype),
                right_face_grad_constraint=jnp.asarray(rg_er).reshape(-1)[0],
            ).face_value(reconstruction=self.reconstruction)

        if bc_er is not None and hasattr(bc_er, "apply_ghost"):
            er_ghost = bc_er.apply_ghost(er_profile)
            return faces_from_cell_centered(er_ghost)

        return make_profile_cell_variable(
            er_profile,
            field.r_grid_half,
            left_face_grad_constraint=jnp.asarray(0.0, dtype=er_profile.dtype),
            right_face_constraint=er_profile[-1],
        ).face_value(reconstruction=self.reconstruction)

    def ap_linear_split(self, state, flux_models, source_models, field, solver_parameters, bc=None, **kwargs):
        """Return diagonal linearization and explicit source for optional AP preconditioning.

        This is an optional helper. It does not alter default solver behavior.
        """
        del source_models
        dr_cells = jnp.diff(field.r_grid_half)
        Vprime = field.Vprime
        Vprime_half = field.Vprime_half

        def diffusion_part(er_vec):
            er_faces = self._build_er_faces(er_vec, field, bc)
            er_diff = conservative_update(er_faces, dr_cells, Vprime, Vprime_half)
            return solver_parameters.Er_relax * solver_parameters.DEr * er_diff

        diag_linear = jnp.diag(jax.jacfwd(diffusion_part)(state.Er))
        full_rhs = self(state, flux_models, None, field, solver_parameters, bc=bc, **kwargs)
        explicit_source = full_rhs - diag_linear * state.Er
        return diag_linear, explicit_source

    def __call__(self, state, flux_models, source_models, field, solver_parameters, bc=None, **kwargs):
        dr_cells = jnp.diff(field.r_grid_half)
        Vprime = field.Vprime
        Vprime_half = field.Vprime_half
        Gamma_total = flux_models.get("Gamma_total")
        Er = state.Er

        bc_gamma = None
        if bc is not None:
            bc_gamma = bc.get("density", bc.get("Er"))

        if bc_gamma is not None and hasattr(bc_gamma, "right_type"):
            rv, rg = right_constraints_from_bc_model(bc_gamma, Gamma_total[:, -1])
            if rv is not None:
                Gamma_faces = jax.vmap(
                    lambda G, right_val: make_profile_cell_variable(
                        G,
                        field.r_grid_half,
                        left_face_constraint=jnp.asarray(0.0, dtype=G.dtype),
                        right_face_constraint=right_val,
                    ).face_value(reconstruction=self.reconstruction)
                )(Gamma_total, jnp.asarray(rv))
            else:
                Gamma_faces = jax.vmap(
                    lambda G, right_grad: make_profile_cell_variable(
                        G,
                        field.r_grid_half,
                        left_face_constraint=jnp.asarray(0.0, dtype=G.dtype),
                        right_face_grad_constraint=right_grad,
                    ).face_value(reconstruction=self.reconstruction)
                )(Gamma_total, jnp.asarray(rg))
        # Backward-compatible path for legacy BC objects.
        elif bc_gamma is not None and hasattr(bc_gamma, "apply_ghost"):
            if hasattr(bc_gamma, "apply_ghost_all"):
                Gamma_ghost = bc_gamma.apply_ghost_all(Gamma_total)
            else:
                Gamma_ghost = jax.vmap(lambda G: bc_gamma.apply_ghost(G))(Gamma_total)
            Gamma_faces = jax.vmap(faces_from_cell_centered)(Gamma_ghost)
        else:
            Gamma_faces = jax.vmap(
                lambda G: make_profile_cell_variable(
                    G,
                    field.r_grid_half,
                    left_face_constraint=jnp.asarray(0.0, dtype=G.dtype),
                    right_face_grad_constraint=jnp.asarray(0.0, dtype=G.dtype),
                ).face_value(reconstruction=self.reconstruction)
            )(Gamma_total)

        Er_faces = self._build_er_faces(Er, field, bc)

        ambipolar_flux_center = 0.5 * (Gamma_faces[:, :-1] + Gamma_faces[:, 1:])
        species_mass = kwargs.get("species_mass")
        charge_qp = kwargs.get("charge_qp")
        plasma_permitivity = jax.vmap(_plasma_permitivity, in_axes=(None, None, None, 0))(state, species_mass, field, field.r_grid)
        ambi_term = jnp.sum(charge_qp[:, None] * ambipolar_flux_center, axis=0) * elementary_charge * 1.e-3 / plasma_permitivity
        Er_diffusion = conservative_update(Er_faces, dr_cells, Vprime, Vprime_half)
        SourceEr = solver_parameters.Er_relax * (solver_parameters.DEr * Er_diffusion - ambi_term)
        SourceEr = SourceEr.at[0].set(0.)
        return SourceEr


