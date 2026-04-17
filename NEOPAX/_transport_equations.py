from typing import Dict, Type
import dataclasses
import jax
import jax.numpy as jnp
from jax import jit
from ._fem import conservative_update, faces_from_cell_centered
from ._cell_variable import make_profile_cell_variable
from ._boundary_conditions import left_constraints_from_bc_model, right_constraints_from_bc_model
from ._constants import elementary_charge

DENSITY_STATE_TO_PHYSICAL = 1.0e20


def enforce_quasi_neutrality(state, species):
    """
    Reconstruct electron density from ion densities and species charges.
    Returns a new TransportState with quasi-neutral electron density.
    """
    from ._species import get_species_idx
    charge_qp = jnp.asarray(species.charge_qp)
    eidx = get_species_idx("e", species.names)
    ion_indices = jnp.array(species.ion_indices)
    Z_i = jnp.take(charge_qp, ion_indices, axis=0)
    n_i = jnp.take(state.density, ion_indices, axis=0)
    Z_e = charge_qp[eidx]
    n_e = -jnp.sum(Z_i[:, None] * n_i, axis=0) / Z_e
    density = state.density.at[eidx, :].set(n_e)
    return dataclasses.replace(state, density=density)

@jit
def _plasma_permitivity_from_prefactor(state, species_mass, permitivity_prefactor):
    """Plasma permittivity on the transport grid using a precomputed geometry prefactor."""
    mass_density = DENSITY_STATE_TO_PHYSICAL * jnp.sum(species_mass[:, None] * state.density, axis=0)
    return mass_density * permitivity_prefactor

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
    """

    def __call__(self, state, models, field, species, energy_grid, database, solver_parameters, bc=None, **kwargs):
        raise NotImplementedError

def get_equation(name: str) -> Type:
    return __equation_registry[name]

def list_equations():
    return list(__equation_registry.keys())

# --- Example built-in equation: Density evolution ---

# --- JAX-friendly, torax-style DensityEquation ---
@register_equation("density")
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class DensityEquation(EquationBase):
    dr_cells: jax.Array = dataclasses.field(repr=False)
    Vprime: jax.Array = dataclasses.field(repr=False)
    Vprime_half: jax.Array = dataclasses.field(repr=False)
    flux_model: callable = dataclasses.field(repr=False)
    gamma_faces_builder: callable = dataclasses.field(repr=False)
    source_model: callable = dataclasses.field(repr=False, default=None)
    name: str = "density"

    def __call__(self, state, fluxes=None):
        if fluxes is None:
            fluxes = self.flux_model(state)
        Gamma = fluxes["Gamma"]
        Gamma_faces = self.gamma_faces_builder(Gamma)
        density_rhs = jax.vmap(
            lambda flux: conservative_update(flux, self.dr_cells, self.Vprime, self.Vprime_half)
        )(Gamma_faces)
        if self.source_model is not None:
            density_rhs += self.source_model(state)
        return density_rhs

# --- Factory function to build DensityEquation up front ---
def build_density_equation(field, flux_model, source_model, bc_density, reconstruction="linear"):
    dr_cells = jnp.diff(field.r_grid_half)
    Vprime = field.Vprime
    Vprime_half = field.Vprime_half
    # Pre-build the gamma_faces_builder function for BC handling
    if bc_density is not None and hasattr(bc_density, "right_type"):
        def gamma_faces_builder(Gamma):
            rv, rg = right_constraints_from_bc_model(bc_density, Gamma[:, -1])
            if rv is not None:
                return jax.vmap(
                    lambda G, right_val: make_profile_cell_variable(
                        G,
                        field.r_grid_half,
                        left_face_constraint=jnp.asarray(0.0, dtype=G.dtype),
                        right_face_constraint=right_val,
                    ).face_value(reconstruction=reconstruction)
                )(Gamma, jnp.asarray(rv))
            else:
                return jax.vmap(
                    lambda G, right_grad: make_profile_cell_variable(
                        G,
                        field.r_grid_half,
                        left_face_constraint=jnp.asarray(0.0, dtype=G.dtype),
                        right_face_grad_constraint=right_grad,
                    ).face_value(reconstruction=reconstruction)
                )(Gamma, jnp.asarray(rg))
    elif bc_density is not None and hasattr(bc_density, "apply_ghost"):
        def gamma_faces_builder(Gamma):
            if hasattr(bc_density, "apply_ghost_all"):
                Gamma_ghost = bc_density.apply_ghost_all(Gamma)
            else:
                Gamma_ghost = jax.vmap(lambda G: bc_density.apply_ghost(G))(Gamma)
            return jax.vmap(faces_from_cell_centered)(Gamma_ghost)
    else:
        def gamma_faces_builder(Gamma):
            return jax.vmap(
                lambda G: make_profile_cell_variable(
                    G,
                    field.r_grid_half,
                    left_face_constraint=jnp.asarray(0.0, dtype=G.dtype),
                    right_face_grad_constraint=jnp.asarray(0.0, dtype=G.dtype),
                ).face_value(reconstruction=reconstruction)
            )(Gamma)
    return DensityEquation(
        dr_cells=dr_cells,
        Vprime=Vprime,
        Vprime_half=Vprime_half,
        flux_model=flux_model,
        source_model=source_model,
        gamma_faces_builder=gamma_faces_builder,
    )

# --- Example built-in equation: Temperature evolution ---

# --- JAX-friendly, torax-style TemperatureEquation ---
@register_equation("temperature")
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class TemperatureEquation(EquationBase):
    dr_cells: jax.Array = dataclasses.field(repr=False)
    Vprime: jax.Array = dataclasses.field(repr=False)
    Vprime_half: jax.Array = dataclasses.field(repr=False)
    flux_model: callable = dataclasses.field(repr=False)
    q_faces_builder: callable = dataclasses.field(repr=False)
    source_model: callable = dataclasses.field(repr=False, default=None)
    name: str = "temperature"

    def __call__(self, state, fluxes=None):
        if fluxes is None:
            fluxes = self.flux_model(state)
        Q = fluxes["Q"]
        Q_faces = self.q_faces_builder(Q)
        temp_rhs = jax.vmap(
            lambda flux: conservative_update(flux, self.dr_cells, self.Vprime, self.Vprime_half)
        )(Q_faces)
        if self.source_model is not None:
            temp_rhs += self.source_model(state)
        return temp_rhs

# --- Factory function to build TemperatureEquation up front ---
def build_temperature_equation(field, flux_model, source_model, bc_temperature, reconstruction="linear"):
    dr_cells = jnp.diff(field.r_grid_half)
    Vprime = field.Vprime
    Vprime_half = field.Vprime_half
    # Pre-build the q_faces_builder function for BC handling
    if bc_temperature is not None and hasattr(bc_temperature, "right_type"):
        def q_faces_builder(Q):
            rv, rg = right_constraints_from_bc_model(bc_temperature, Q[:, -1])
            if rv is not None:
                return jax.vmap(
                    lambda Qv, right_val: make_profile_cell_variable(
                        Qv,
                        field.r_grid_half,
                        left_face_constraint=jnp.asarray(0.0, dtype=Qv.dtype),
                        right_face_constraint=right_val,
                    ).face_value(reconstruction=reconstruction)
                )(Q, jnp.asarray(rv))
            else:
                return jax.vmap(
                    lambda Qv, right_grad: make_profile_cell_variable(
                        Qv,
                        field.r_grid_half,
                        left_face_constraint=jnp.asarray(0.0, dtype=Qv.dtype),
                        right_face_grad_constraint=right_grad,
                    ).face_value(reconstruction=reconstruction)
                )(Q, jnp.asarray(rg))
    elif bc_temperature is not None and hasattr(bc_temperature, "apply_ghost"):
        def q_faces_builder(Q):
            if hasattr(bc_temperature, "apply_ghost_all"):
                Q_ghost = bc_temperature.apply_ghost_all(Q)
            else:
                Q_ghost = jax.vmap(lambda Qv: bc_temperature.apply_ghost(Qv))(Q)
            return jax.vmap(faces_from_cell_centered)(Q_ghost)
    else:
        def q_faces_builder(Q):
            return jax.vmap(
                lambda Qv: make_profile_cell_variable(
                    Qv,
                    field.r_grid_half,
                    left_face_constraint=jnp.asarray(0.0, dtype=Qv.dtype),
                    right_face_grad_constraint=jnp.asarray(0.0, dtype=Qv.dtype),
                ).face_value(reconstruction=reconstruction)
            )(Q)
    return TemperatureEquation(
        dr_cells=dr_cells,
        Vprime=Vprime,
        Vprime_half=Vprime_half,
        flux_model=flux_model,
        source_model=source_model,
        q_faces_builder=q_faces_builder,
    )

# --- Example built-in equation: Electric field (Er) evolution ---

# --- JAX-friendly, torax-style ElectricFieldEquation ---
@register_equation("Er")
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class ElectricFieldEquation(EquationBase):
    dr_cells: jax.Array = dataclasses.field(repr=False)
    Vprime: jax.Array = dataclasses.field(repr=False)
    Vprime_half: jax.Array = dataclasses.field(repr=False)
    flux_model: callable = dataclasses.field(repr=False)
    species_mass: jax.Array = dataclasses.field(repr=False)
    charge_qp: jax.Array = dataclasses.field(repr=False)
    permitivity_prefactor: jax.Array = dataclasses.field(repr=False)
    gamma_faces_builder: callable = dataclasses.field(repr=False)
    er_diffusive_flux_builder: callable = dataclasses.field(repr=False)
    source_mode: str = "ambipolar_local"
    Er_relax: float = 1.0
    DEr: float = 1.0
    boundary_mode: str = "standard"
    Er_edge_relax: float = 1.0
    name: str = "Er"

    def _charge_flux_from_gamma(self, Gamma):
        mode = str(self.source_mode).strip().lower()
        if mode in {"ambipolar_local", "transport_local", "local"}:
            return jnp.sum(self.charge_qp[:, None] * Gamma, axis=0)

        Gamma_faces = self.gamma_faces_builder(Gamma)
        ambipolar_flux_center = 0.5 * (Gamma_faces[:, :-1] + Gamma_faces[:, 1:])
        return jnp.sum(self.charge_qp[:, None] * ambipolar_flux_center, axis=0)

    def _er_diffusion(self, Er):
        # When DEr == 0 we want a true pure-ambipolar RHS, not 0 * NaN.
        if float(self.DEr) == 0.0:
            er_diffusive_flux = jnp.zeros(Er.shape[0] + 1, dtype=Er.dtype)
            er_diffusion = jnp.zeros_like(Er)
        else:
            er_diffusive_flux = self.er_diffusive_flux_builder(Er)
            er_diffusion = conservative_update(
                er_diffusive_flux, self.dr_cells, self.Vprime, self.Vprime_half
            )
        return er_diffusive_flux, er_diffusion

    def debug_components(self, state, fluxes=None):
        if fluxes is None:
            fluxes = self.flux_model(state)
        Er = state.Er
        plasma_permitivity = _plasma_permitivity_from_prefactor(
            state,
            self.species_mass,
            self.permitivity_prefactor,
        )
        Gamma = fluxes["Gamma"]
        charge_flux = self._charge_flux_from_gamma(Gamma)
        ambi_term = charge_flux * elementary_charge * 1.e-3 / plasma_permitivity
        er_diffusive_flux, er_diffusion = self._er_diffusion(Er)
        return {
            "charge_flux": charge_flux,
            "plasma_permitivity": plasma_permitivity,
            "ambi_term": ambi_term,
            "er_diffusive_flux": er_diffusive_flux,
            "er_diffusion": er_diffusion,
        }

    def __call__(self, state, fluxes=None):
        if fluxes is None:
            fluxes = self.flux_model(state)
        Er = state.Er
        plasma_permitivity = _plasma_permitivity_from_prefactor(
            state,
            self.species_mass,
            self.permitivity_prefactor,
        )
        Gamma = fluxes["Gamma"]
        charge_flux = self._charge_flux_from_gamma(Gamma)
        ambi_term = charge_flux * elementary_charge * 1.e-3 / plasma_permitivity
        _, Er_diffusion = self._er_diffusion(Er)
        SourceEr = self.Er_relax * (self.DEr * Er_diffusion - ambi_term)
        SourceEr = SourceEr.at[0].set(0.)
        if self.boundary_mode == "floating_ambipolar_edge":
            SourceEr = SourceEr.at[-1].set(-self.Er_edge_relax * ambi_term[-1])
        return SourceEr

    def ap_linear_split(self, state):
        """
        Return diagonal linearization and explicit source for optional AP preconditioning.
        Uses only attributes set at construction and the current state.
        """
        def diffusion_part(er_vec):
            er_diffusive_flux = self.er_diffusive_flux_builder(er_vec)
            er_diff = conservative_update(er_diffusive_flux, self.dr_cells, self.Vprime, self.Vprime_half)
            return self.Er_relax * self.DEr * er_diff

        diag_linear = jnp.diag(jax.jacfwd(diffusion_part)(state.Er))
        full_rhs = self(state)
        explicit_source = full_rhs - diag_linear * state.Er
        return diag_linear, explicit_source

# --- Factory function to build ElectricFieldEquation up front ---
def build_electric_field_equation(
    field,
    flux_model,
    species_mass,
    charge_qp,
    bc_gamma,
    bc_er,
    Er_relax=1.0,
    DEr=1.0,
    source_mode="ambipolar_local",
    reconstruction="linear",
    boundary_mode="standard",
    Er_edge_relax=1.0,
):
    dr_cells = jnp.diff(field.r_grid_half)
    Vprime = field.Vprime
    Vprime_half = field.Vprime_half
    psi_fac = 1.0 + 1.0 / (field.enlogation * jnp.square(field.iota))
    psi_fac = psi_fac.at[0].set(1.0)
    permitivity_prefactor = psi_fac / jnp.square(field.B0)
    # Pre-build the gamma_faces_builder function for BC handling (density/Er)
    if bc_gamma is not None and hasattr(bc_gamma, "right_type"):
        def gamma_faces_builder(Gamma):
            rv, rg = right_constraints_from_bc_model(bc_gamma, Gamma[:, -1])
            if rv is not None:
                return jax.vmap(
                    lambda G, right_val: make_profile_cell_variable(
                        G,
                        field.r_grid_half,
                        left_face_constraint=jnp.asarray(0.0, dtype=G.dtype),
                        right_face_constraint=right_val,
                    ).face_value(reconstruction=reconstruction)
                )(Gamma, jnp.asarray(rv))
            else:
                return jax.vmap(
                    lambda G, right_grad: make_profile_cell_variable(
                        G,
                        field.r_grid_half,
                        left_face_constraint=jnp.asarray(0.0, dtype=G.dtype),
                        right_face_grad_constraint=right_grad,
                    ).face_value(reconstruction=reconstruction)
                )(Gamma, jnp.asarray(rg))
    elif bc_gamma is not None and hasattr(bc_gamma, "apply_ghost"):
        def gamma_faces_builder(Gamma):
            if hasattr(bc_gamma, "apply_ghost_all"):
                Gamma_ghost = bc_gamma.apply_ghost_all(Gamma)
            else:
                Gamma_ghost = jax.vmap(lambda G: bc_gamma.apply_ghost(G))(Gamma)
            return jax.vmap(faces_from_cell_centered)(Gamma_ghost)
    else:
        def gamma_faces_builder(Gamma):
            return jax.vmap(
                lambda G: make_profile_cell_variable(
                    G,
                    field.r_grid_half,
                    left_face_constraint=jnp.asarray(0.0, dtype=G.dtype),
                    right_face_grad_constraint=jnp.asarray(0.0, dtype=G.dtype),
                ).face_value(reconstruction=reconstruction)
            )(Gamma)
    # Pre-build the diffusive Er face-flux builder for BC handling.
    if bc_er is not None and hasattr(bc_er, "right_type"):
        def er_diffusive_flux_builder(er_profile):
            lv_er, lg_er = left_constraints_from_bc_model(bc_er, er_profile[0])
            rv_er, rg_er = right_constraints_from_bc_model(bc_er, er_profile[-1])
            if rv_er is not None:
                er_cell_var = make_profile_cell_variable(
                    er_profile,
                    field.r_grid_half,
                    left_face_constraint=None if lv_er is None else jnp.asarray(lv_er).reshape(-1)[0],
                    left_face_grad_constraint=None if lg_er is None else jnp.asarray(lg_er).reshape(-1)[0],
                    right_face_constraint=jnp.asarray(rv_er).reshape(-1)[0],
                )
            else:
                er_cell_var = make_profile_cell_variable(
                er_profile,
                field.r_grid_half,
                    left_face_constraint=None if lv_er is None else jnp.asarray(lv_er).reshape(-1)[0],
                    left_face_grad_constraint=None if lg_er is None else jnp.asarray(lg_er).reshape(-1)[0],
                    right_face_grad_constraint=jnp.asarray(rg_er).reshape(-1)[0],
                )
            return -er_cell_var.face_grad()
    elif bc_er is not None and hasattr(bc_er, "apply_ghost"):
        def er_diffusive_flux_builder(er_profile):
            er_ghost = bc_er.apply_ghost(er_profile)
            return -jnp.diff(er_ghost) / jnp.diff(field.r_grid_half)
    else:
        def er_diffusive_flux_builder(er_profile):
            er_cell_var = make_profile_cell_variable(
                er_profile,
                field.r_grid_half,
                left_face_grad_constraint=jnp.asarray(0.0, dtype=er_profile.dtype),
                right_face_constraint=er_profile[-1],
            )
            return -er_cell_var.face_grad()
    return ElectricFieldEquation(
        dr_cells=dr_cells,
        Vprime=Vprime,
        Vprime_half=Vprime_half,
        flux_model=flux_model,
        species_mass=species_mass,
        charge_qp=charge_qp,
        permitivity_prefactor=permitivity_prefactor,
        gamma_faces_builder=gamma_faces_builder,
        er_diffusive_flux_builder=er_diffusive_flux_builder,
        source_mode=str(source_mode).strip().lower(),
        Er_relax=Er_relax,
        DEr=DEr,
        boundary_mode=str(boundary_mode).strip().lower(),
        Er_edge_relax=Er_edge_relax,
    )


# --- Equation System Builder (torax-style) ---
def build_equation_system(
    config,
    species,
    field,
    flux_model,
    source_models=None,
    solver_cfg=None,
    boundary_models=None,
):
    """
    Build the list of equation instances to evolve using prebuilt runtime
    objects. This avoids rebuilding geometry, databases, and flux models inside
    the equation builder and keeps compile closures smaller.
    """
    equations_cfg = config.get("equations", {})
    eqn_flags = {
        "density": equations_cfg.get("toggle_density", [True]*getattr(species, 'number_species', 3)),
        "temperature": equations_cfg.get("toggle_temperature", [True]*getattr(species, 'number_species', 3)),
        "Er": equations_cfg.get("toggle_Er", True),
    }
    equations_to_evolve = []
    species_mass = getattr(species, "mass", None)
    charge_qp = getattr(species, "charge_qp", None)
    solver_cfg = {} if solver_cfg is None else solver_cfg
    boundary_models = {} if boundary_models is None else boundary_models
    source_models = {} if source_models is None else source_models
    bc_density = boundary_models.get("density")
    bc_temperature = boundary_models.get("temperature")
    bc_gamma = boundary_models.get("gamma")
    bc_er = boundary_models.get("Er")
    density_source_model = source_models.get("density")
    temperature_source_model = source_models.get("temperature")
    Er_relax = solver_cfg.get("Er_relax", 1.0)
    DEr = solver_cfg.get("DEr", 1.0)
    Er_source_mode = solver_cfg.get("Er_source_mode", "transport_centered")
    Er_boundary_mode = solver_cfg.get("Er_right_boundary_mode", solver_cfg.get("Er_boundary_mode", "standard"))
    Er_edge_relax = solver_cfg.get("Er_edge_relax", Er_relax)

    if any(eqn_flags["density"]):
        equations_to_evolve.append(build_density_equation(
            field,
            flux_model,
            density_source_model,
            bc_density,
        ))
    if any(eqn_flags["temperature"]):
        equations_to_evolve.append(build_temperature_equation(
            field,
            flux_model,
            temperature_source_model,
            bc_temperature,
        ))
    if eqn_flags["Er"]:
        equations_to_evolve.append(build_electric_field_equation(
            field,
            flux_model,
            species_mass,
            charge_qp,
            bc_gamma,
            bc_er,
            Er_relax=Er_relax,
            DEr=DEr,
            source_mode=Er_source_mode,
            boundary_mode=Er_boundary_mode,
            Er_edge_relax=Er_edge_relax,
        ))
    return equations_to_evolve


def build_equation_system_from_config(config, species):
    """
    Backward-compatible wrapper that builds the required runtime objects from
    config before delegating to ``build_equation_system``.
    """
    from ._boundary_conditions import build_boundary_condition_model
    from ._database import Monoenergetic
    from ._energy_grid_models import get_energy_grid_model
    from ._geometry_models import get_geometry_model
    from ._source_models import build_source_models_from_config
    from ._transport_flux_models import ZeroTransportModel, build_transport_flux_model, get_transport_flux_model

    geom_cfg = config.get("geometry", {})
    n_radial = int(geom_cfg.get("n_radial", 51))
    vmec_file = geom_cfg.get("vmec_file")
    boozer_file = geom_cfg.get("boozer_file")
    field = None
    if vmec_file is not None and boozer_file is not None:
        field = get_geometry_model("vmec_booz", n_r=n_radial, vmec=vmec_file, booz=boozer_file)

    energy_grid_cfg = config.get("energy_grid", {})
    n_x = int(energy_grid_cfg.get("n_x", 4))
    energy_grid = get_energy_grid_model("standard_laguerre", n_x=n_x, n_order=3)
    neoclassical_cfg = config.get("neoclassical", {})
    database = None
    neoclassical_file = neoclassical_cfg.get("neoclassical_file")
    if neoclassical_file and field is not None:
        database = Monoenergetic.read_monkes(field.a_b, neoclassical_file)

    neoclassical_factory = get_transport_flux_model(neoclassical_cfg.get("flux_model", "monkes_database"))
    turbulence_factory = get_transport_flux_model(config.get("turbulence", {}).get("flux_model", "none"))
    classical_factory = get_transport_flux_model(config.get("classical", {}).get("flux_model", "none")) if "classical" in config else None
    neoclassical_model = neoclassical_factory(species, energy_grid, field, database)
    turbulence_model = turbulence_factory(species, energy_grid, field, database) if turbulence_factory is not None else ZeroTransportModel()
    classical_model = classical_factory(species, energy_grid, field, database) if classical_factory is not None else ZeroTransportModel()
    flux_model = build_transport_flux_model(neoclassical_model, turbulence_model, classical_model)

    boundary_cfg = config.get("boundary", {})
    dr = getattr(field, "dr", 1.0)
    boundary_models = {
        key: build_boundary_condition_model(boundary_cfg[key], dr)
        for key in ("density", "temperature", "Er", "gamma")
        if key in boundary_cfg
    }
    solver_cfg = config.get("transport_solver", {})
    if not solver_cfg:
        solver_cfg = config.get("solver", config.get("transport", {}))
    source_models = build_source_models_from_config(config)

    return build_equation_system(
        config=config,
        species=species,
        field=field,
        flux_model=flux_model,
        source_models=source_models,
        solver_cfg=solver_cfg,
        boundary_models=boundary_models,
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class ComposedEquationSystem:
    equations: tuple
    species: object | None = None
    shared_flux_model: object | None = None

    def __call__(self, t, state, runtime):
        """
        Call all equations with state, return a TransportState matching the state structure.
        Always output all three fields, setting missing ones to zero arrays of the correct shape.
        For density, always output a zero array for electrons if quasi-neutrality is enforced.
        """
        import jax.numpy as jnp
        from ._state import TransportState
        from ._species import get_species_idx
        shared_fluxes = None
        if self.shared_flux_model is not None:
            shared_fluxes = self.shared_flux_model(state)
        # Map equation types to their outputs
        eq_outputs = {}
        for eq in self.equations:
            name = getattr(eq, 'name', None)
            if name is not None:
                eq_outputs[name] = eq(state, fluxes=shared_fluxes)

        # Defensive: always use shape of state.density, state.temperature, state.Er
        density_rhs = eq_outputs.get('density', jnp.zeros_like(state.density))
        temperature_rhs = eq_outputs.get('temperature', jnp.zeros_like(state.temperature))
        Er_rhs = eq_outputs.get('Er', jnp.zeros_like(state.Er))

        # If density_rhs is missing, fill with zeros
        if density_rhs.shape != state.density.shape:
            density_rhs = jnp.zeros_like(state.density)

        # Always set electron density_rhs to zero (quasi-neutrality enforced elsewhere)
        try:
            eidx = get_species_idx("e", getattr(self.species, 'names', ()))
            density_rhs = density_rhs.at[eidx, :].set(0.0)
        except Exception:
            pass

        return TransportState(
            density=density_rhs,
            temperature=temperature_rhs,
            Er=Er_rhs,
        )

    def vector_field(self, t, y, args):
        """
        Torax-style vector field for JAX ODE solvers: (t, y, args) -> dy/dt
        y is the state, args[0] is the runtime dict.
        """
        return self(t,y, args)


