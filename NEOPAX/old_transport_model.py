import jax
from jax import config, jit
# to use higher precision
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import interpax
from ._constants import Boltzmann, elementary_charge, epsilon_0, hbar, proton_mass
import diffrax
import jax.experimental.host_callback as hcb
from ._species import Species
from ._neoclassical import get_Neoclassical_Fluxes,get_Neoclassical_Fluxes_With_Momentum_Correction
from ._physics import get_plasma_permitivity, Power_Exchange, get_DT_Reaction, FusionPowerFractionElectrons, P_rad
from ._fem import conservative_update, faces_from_cell_centered
from ._fem import set_dirichlet_ghosts, set_neumann_ghosts
import dataclasses

@dataclasses.dataclass(frozen=True)
class TransportState:
    density: jnp.ndarray  # shape (n_species, n_radial)
    temperature: jnp.ndarray  # shape (n_species, n_radial)
    Er: jnp.ndarray  # shape (n_radial,)
    # Add more fields as needed (e.g., impurity densities, etc.)

# --- Modular FVM, Source, and BC Logic ---

class FVMScheme:
    def __init__(self, field):
        self.field = field
        self.dr = field.dr
        self.Vprime = field.Vprime
        self.Vprime_half = field.Vprime_half

    def conservative_update(self, flux):
        return conservative_update(flux, self.dr, self.Vprime, self.Vprime_half)

    def extend_with_ghosts(self, arr):
        return jnp.concatenate([arr[:1], arr, arr[-1:]])

    def apply_dirichlet_ghosts(self, arr):
        arr_ext = self.extend_with_ghosts(arr)
        return set_dirichlet_ghosts(arr_ext, arr[0], arr[-1])

    def faces_from_cell_centered(self, arr):
        return faces_from_cell_centered(arr)



# Standard JAX-compatible interface for source/transport models
class SourceModel:
    def __init__(self, field, database, turbulent, solver_parameters):
        self.field = field
        self.database = database
        self.turbulent = turbulent
        self.solver_parameters = solver_parameters

    def __call__(self, state: TransportState):
        dr = self.field.dr
        n_species, n_radial = state.density.shape
        Er = state.Er
        # Select neoclassical flux function based on momentum_correction_flag
        if getattr(self.solver_parameters, 'momentum_correction_flag', False):
            _, Gamma_neo, Q_neo, Upar = get_Neoclassical_Fluxes_With_Momentum_Correction(state, self.field, self.field, self.database, Er)
        else:
            _, Gamma_neo, Q_neo, Upar = get_Neoclassical_Fluxes(state, self.field, self.field, self.database, Er)
        Gamma_turb = self.turbulent.Gamma_turb if hasattr(self.turbulent, 'Gamma_turb') else jnp.zeros_like(Gamma_neo)
        Q_turb = self.turbulent.Q_turb if hasattr(self.turbulent, 'Q_turb') else jnp.zeros_like(Q_neo)
        Gamma_total = Gamma_neo + Gamma_turb
        Q_total = Q_neo + Q_turb

        # Vectorized power exchange: shape [n_species, n_species, n_radial]
        power_exchange = jax.vmap(
            lambda i: jax.vmap(
                lambda j: jax.vmap(lambda r: Power_Exchange(state, i, j, r))(jnp.arange(n_radial))
            )(jnp.arange(n_species))
        )(jnp.arange(n_species))

        # Vectorized fusion: get_DT_Reaction returns (rate, HeSource, AlphaPower)
        DTreactionRate, HeSource, AlphaPower = jax.vmap(lambda r: get_DT_Reaction(state, r))(jnp.arange(n_radial))
        fusion_power = jnp.zeros((n_species, n_radial))
        fusion_power = fusion_power.at[1].set(0.5 * AlphaPower)  # D
        fusion_power = fusion_power.at[2].set(0.5 * AlphaPower)  # T

        # Vectorized bremsstrahlung: shape [n_radial]
        PBrems, Zeff = jax.vmap(lambda r: P_rad(state, r))(jnp.arange(n_radial))
        bremsstrahlung = jnp.zeros((n_species, n_radial))
        bremsstrahlung = bremsstrahlung.at[0].set(PBrems)  # electrons

        def extend_with_ghosts(u):
            return jnp.concatenate([u[:1], u, u[-1:]])
        density_ghost = jax.vmap(lambda n: set_dirichlet_ghosts(extend_with_ghosts(n), n[0], n[-1]))(state.density)
        temp_ghost = jax.vmap(lambda T: set_dirichlet_ghosts(extend_with_ghosts(T), T[0], T[-1]))(state.temperature)
        Q_ghost = jax.vmap(lambda Q: extend_with_ghosts(Q))(Q_total)
        Gamma_ghost = jax.vmap(lambda G: extend_with_ghosts(G))(Gamma_total)
        Er_ghost = set_dirichlet_ghosts(extend_with_ghosts(Er), Er[0], Er[-1])

        Q_faces = jax.vmap(faces_from_cell_centered)(Q_ghost)
        Gamma_faces = jax.vmap(faces_from_cell_centered)(Gamma_ghost)
        Er_faces = faces_from_cell_centered(Er_ghost)

        density_rhs = jax.vmap(lambda flux: conservative_update(flux, dr, self.field.Vprime, self.field.Vprime_half))(Gamma_faces)
        temp_rhs = jax.vmap(lambda flux: conservative_update(flux, dr, self.field.Vprime, self.field.Vprime_half))(Q_faces)
        temp_rhs = temp_rhs + jnp.sum(power_exchange, axis=1) + fusion_power - bremsstrahlung
        temperature_rhs = temp_rhs

        er_mode = getattr(self.solver_parameters, 'er_mode', 'diffusion')
        if er_mode == 'diffusion':
            ambipolar_flux_center = 0.5 * (Gamma_faces[:, :-1] + Gamma_faces[:, 1:])
            plasma_permitivity = jax.vmap(get_plasma_permitivity, in_axes=(None, None, 0))(state, self.field, self.field.r_grid)
            ambi_term = jnp.sum(state.charge_qp[:, None] * ambipolar_flux_center, axis=0) * elementary_charge * 1.e-3 / plasma_permitivity
            Er_diffusion = conservative_update(Er_faces, dr, self.field.Vprime, self.field.Vprime_half)
            er_on = 1.0 if bool(getattr(self.solver_parameters, "evolve_Er", True)) else 0.0
            SourceEr = er_on * self.solver_parameters.Er_relax * (self.solver_parameters.DEr * Er_diffusion - ambi_term)
            SourceEr = SourceEr.at[0].set(0.)
        else:
            SourceEr = jnp.zeros_like(Er)

        return temperature_rhs, density_rhs, SourceEr


@jit
def vector_field(t, state: TransportState, args):
    Initial_Species, grid, field, database, turbulent, solver_parameters = args
    # For entropy mode, compute Er from root-finder, ignore Er from state
    if getattr(solver_parameters, 'er_mode', 'diffusion') == 'entropy':
        # Build a temporary Species object for root-finding
        species_tmp = Species(
            Initial_Species.number_species,
            Initial_Species.radial_points,
            Initial_Species.species_indeces,
            Initial_Species.mass_mp,
            Initial_Species.charge_qp,
            temperature,
            density,
            Er,  # This Er is just a placeholder, will be replaced
            field.r_grid,
            field.r_grid_half,
            field.dr,
            field.Vprime_half,
            field.overVprime,
            Initial_Species.n_edge,
            Initial_Species.T_edge
        )
        from .ambipolarity import find_ambipolar_Er_all_roots
        # Select neoclassical flux function for root-finding
        if getattr(solver_parameters, 'momentum_correction_flag', False):
            flux_func = get_Neoclassical_Fluxes_With_Momentum_Correction
        else:
            flux_func = get_Neoclassical_Fluxes
        def Gamma_func(i, Er_val):
            Er_vec = Er.at[i].set(Er_val)
            _, Gamma_neo, _, _ = flux_func(species_tmp, grid, field, database, Er_vec)
            return jnp.sum(Initial_Species.charge_qp * Gamma_neo[:, i])
        def entropy_func(i, Er_val):
            Er_vec = Er.at[i].set(Er_val)
            _, Gamma_neo, _, _ = flux_func(species_tmp, grid, field, database, Er_vec)
            return jnp.sum(jnp.abs(Gamma_neo[:, i]))
        def find_best_Er(i):
            gamma = lambda Er_val: Gamma_func(i, Er_val)
            entropy = lambda Er_val: entropy_func(i, Er_val)
            best_Er, _, _ = find_ambipolar_Er_all_roots(gamma, entropy)
            return best_Er
        Er_entropy = jax.vmap(find_best_Er)(jnp.arange(nrad))
        Er = Er_entropy  # Use the root-finder value for all physics
        dEr_dt = jnp.zeros_like(Er)
    else:
        dEr_dt = None  # Will be set by sources

    # Build the Species object for sources
    species_new = Species(
        Initial_Species.number_species,
        Initial_Species.radial_points,
        Initial_Species.species_indeces,
        Initial_Species.mass_mp,
        Initial_Species.charge_qp,
        temperature,
        density,
        Er,
        field.r_grid,
        field.r_grid_half,
        field.dr,
        field.Vprime_half,
        field.overVprime,
        Initial_Species.n_edge,
        Initial_Species.T_edge
    )
    SourceEr, density_rhs, temperature_rhs = sources(species_new, Er, grid, field, database, turbulent, solver_parameters)
    if dEr_dt is None:
        dEr_dt = SourceEr
    # Return flattened derivatives, with dEr_dt last
    return jnp.concatenate([temperature_rhs.flatten(), density_rhs.flatten(), dEr_dt])





def solve_transport_equations(state0: TransportState, args, model):
    solver_parameters = args[-1]
    term = diffrax.ODETerm(lambda t, y, args: model(y))
    saveat = diffrax.SaveAt(ts=solver_parameters.ts_list)
    stepsize_controller = diffrax.PIDController(pcoeff=0.3, icoeff=0.4, rtol=solver_parameters.rtol, atol=solver_parameters.atol, dtmax=None, dtmin=None)
    solver = diffrax.Kvaerno5()
    sol = diffrax.diffeqsolve(
        term,
        solver,
        solver_parameters.t0,
        solver_parameters.t_final,
        solver_parameters.dt,
        state0,
        args=args,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=None,
    )
    return sol

# --- Model and BC Composition ---

class MainTransportModel:
    def __init__(self, fvm_scheme, source_model, bc_model):
        self.fvm = fvm_scheme
        self.source = source_model
        # bc_model should be a dict: {'density': bc_obj, 'temperature': bc_obj, 'Er': bc_obj}
        self.bc = bc_model

    def __call__(self, state: TransportState):
        # Apply BCs to each field using the modular BC objects
        density_bc = self.bc['density'].apply(state.density)
        temperature_bc = self.bc['temperature'].apply(state.temperature)
        Er_bc = self.bc['Er'].apply(state.Er)
        # Compute sources (to be filled in with actual logic)
        # sources = self.source.compute(state)
        # For now, just return state with BCs applied
        return TransportState(
            density=density_bc,
            temperature=temperature_bc,
            Er=Er_bc
        )


