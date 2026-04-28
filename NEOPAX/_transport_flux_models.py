from __future__ import annotations


from typing import Any, Callable
import abc
import dataclasses
import h5py
import jax
import jax.numpy as jnp
import interpax
from ._cell_variable import (
    get_gradient_density,
    get_gradient_temperature,
    make_profile_cell_variable,
)
from ._boundary_conditions import (
    left_constraints_from_bc_model,
    right_constraints_from_bc_model,
)
from ._neoclassical import (
    get_Lij_matrix_local,
    get_Neoclassical_Fluxes,
    get_Neoclassical_Fluxes_Faces,
    get_Neoclassical_Fluxes_With_Momentum_Correction,
)
from ._species import get_Thermodynamical_Forces_A1, get_Thermodynamical_Forces_A2, get_Thermodynamical_Forces_A3
from ._state import (
    DEFAULT_TRANSPORT_DENSITY_FLOOR,
    DEFAULT_TRANSPORT_TEMPERATURE_FLOOR,
    TransportState,
    apply_transport_density_floor,
    apply_transport_temperature_floor,
    get_v_thermal,
    safe_density,
    safe_temperature,
)
from ._source_models import assemble_pressure_source_components, sum_source_components

DENSITY_STATE_TO_PHYSICAL = 1.0e20
TEMPERATURE_STATE_TO_PHYSICAL = 1.0e3
PRESSURE_SOURCE_STATE_TO_MW_M3 = 1.0 / 62.422
from ._turbulence import get_Turbulent_Fluxes_Analytical, get_Turbulent_Fluxes_PowerOverN


def compute_total_power_mw(state, species, pressure_source_model, geometry, fallback_mw=3.0):
    fallback = jnp.asarray(fallback_mw, dtype=state.density.dtype)
    if pressure_source_model is None or geometry is None:
        return fallback
    raw_sources = pressure_source_model(state)
    if not isinstance(raw_sources, dict):
        return fallback

    net_power_density = None
    alpha_power = raw_sources.get("AlphaPower")
    if alpha_power is not None:
        net_power_density = jnp.asarray(alpha_power, dtype=state.density.dtype)

    pbrems = raw_sources.get("PBrems")
    if pbrems is not None:
        pbrems_arr = jnp.asarray(pbrems, dtype=state.density.dtype)
        net_power_density = -pbrems_arr if net_power_density is None else net_power_density - pbrems_arr

    for key in ("heating", "external_heating", "ecrh", "icrh", "nbi", "ohmic_heating"):
        value = raw_sources.get(key)
        if value is None:
            continue
        arr = jnp.asarray(value, dtype=state.density.dtype)
        net_power_density = arr if net_power_density is None else net_power_density + arr

    if net_power_density is None:
        components = assemble_pressure_source_components(raw_sources, state, species)
        if not components:
            return fallback
        power_density_state = jnp.sum(sum_source_components(components, state.pressure), axis=0)
        power_density_mw_m3 = PRESSURE_SOURCE_STATE_TO_MW_M3 * power_density_state
        total_power = jnp.trapezoid(power_density_mw_m3 * geometry.Vprime, x=geometry.r_grid)
        return jnp.where(total_power < 0.0, fallback, total_power)

    power_density_mw_m3 = PRESSURE_SOURCE_STATE_TO_MW_M3 * net_power_density
    total_power = jnp.trapezoid(power_density_mw_m3 * geometry.Vprime, x=geometry.r_grid)
    return jnp.where(total_power < 0.0, fallback, total_power)


def compute_total_power_breakdown_mw(state, pressure_source_model, geometry):
    if pressure_source_model is None or geometry is None:
        return {}
    raw_sources = pressure_source_model(state)
    if not isinstance(raw_sources, dict):
        return {}

    breakdown: dict[str, jax.Array] = {}
    dtype = state.density.dtype

    def _integrate_state_power(name, value, sign=1.0):
        if value is None:
            return
        arr = jnp.asarray(value, dtype=dtype)
        power_density_mw_m3 = PRESSURE_SOURCE_STATE_TO_MW_M3 * (jnp.asarray(sign, dtype=dtype) * arr)
        breakdown[name] = jnp.trapezoid(power_density_mw_m3 * geometry.Vprime, x=geometry.r_grid)

    _integrate_state_power("alpha_power_mw", raw_sources.get("AlphaPower"), sign=1.0)
    _integrate_state_power("bremsstrahlung_mw", raw_sources.get("PBrems"), sign=-1.0)

    for key in ("heating", "external_heating", "ecrh", "icrh", "nbi", "ohmic_heating"):
        value = raw_sources.get(key)
        if value is None:
            continue
        _integrate_state_power(f"{key}_mw", value, sign=1.0)

    if breakdown:
        total = jnp.asarray(0.0, dtype=dtype)
        for value in breakdown.values():
            total = total + jnp.asarray(value, dtype=dtype)
        breakdown["net_total_mw"] = total
    return breakdown



# Registry for modular selection
TRANSPORT_FLUX_MODEL_REGISTRY: dict[str, Callable[[], "TransportFluxModelBase"]] = {}

def register_transport_flux_model(name: str, builder: Callable[..., "TransportFluxModelBase"]) -> None:
    TRANSPORT_FLUX_MODEL_REGISTRY[str(name).strip().lower()] = builder

def get_transport_flux_model(name: str) -> Callable[..., "TransportFluxModelBase"]:
    key = str(name).strip().lower()
    if key not in TRANSPORT_FLUX_MODEL_REGISTRY:
        raise ValueError(f"Unknown transport flux model '{name}'.")
    return TRANSPORT_FLUX_MODEL_REGISTRY[key]


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

        def build_local_particle_flux_evaluator(self, state):
                del state
                return None

        def evaluate_face_fluxes(self, state, face_state, **kwargs):
                del state, face_state, kwargs
                return None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class FaceTransportState:
    density: jax.Array
    pressure: jax.Array
    Er: jax.Array

    @property
    def temperature(self):
        return self.pressure / safe_density(self.density)


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
        robin_grad = -default_value / (right_decay + 1e-12)
        return default_value, robin_grad
    return default_value, default_grad


def _extract_face_constraints(
    bc_model: Any,
    state_arr: jax.Array,
    face_centers: jax.Array,
) -> tuple[jax.Array | None, jax.Array | None, jax.Array | None, jax.Array | None]:
    default_left = state_arr[:, 0]
    default_right = state_arr[:, -1]
    left_value, left_grad = left_constraints_from_bc_model(
        bc_model,
        default_left,
        profile=state_arr,
        face_centers=face_centers,
    )
    right_value, right_grad = right_constraints_from_bc_model(
        bc_model,
        default_right,
        profile=state_arr,
        face_centers=face_centers,
    )
    return left_value, left_grad, right_value, right_grad


def _face_profile(profile, face_centers, bc_model=None, reconstruction="linear"):
    if profile.ndim == 1:
        profile_2d = profile[None, :]
        squeeze = True
    else:
        profile_2d = profile
        squeeze = False
    left_value, left_grad, right_value, right_grad = _extract_face_constraints(
        bc_model,
        profile_2d,
        face_centers,
    )
    faces = jax.vmap(
        lambda prof, lv, lg, rv, rg: make_profile_cell_variable(
            prof,
            face_centers,
            left_face_constraint=lv,
            left_face_grad_constraint=lg,
            right_face_constraint=rv,
            right_face_grad_constraint=rg,
        ).face_value(reconstruction=reconstruction)
    )(profile_2d, left_value, left_grad, right_value, right_grad)
    if squeeze:
        return faces[0]
    return faces


def _face_profile_gradient(profile, face_centers, bc_model=None):
    if profile.ndim == 1:
        profile_2d = profile[None, :]
        squeeze = True
    else:
        profile_2d = profile
        squeeze = False
    left_value, left_grad, right_value, right_grad = _extract_face_constraints(
        bc_model,
        profile_2d,
        face_centers,
    )

    grads = jax.vmap(
        lambda prof, lv, lg, rv, rg: make_profile_cell_variable(
            prof,
            face_centers,
            left_face_constraint=lv,
            left_face_grad_constraint=lg,
            right_face_constraint=rv,
            right_face_grad_constraint=rg,
        ).face_grad()
    )(profile_2d, left_value, left_grad, right_value, right_grad)
    if squeeze:
        return grads[0]
    return grads


def build_face_transport_state(
    state: TransportState,
    geometry: Any,
    *,
    bc_density: Any = None,
    bc_temperature: Any = None,
    bc_er: Any = None,
    reconstruction: str = "linear",
    density_floor: Any = DEFAULT_TRANSPORT_DENSITY_FLOOR,
    temperature_floor: Any = DEFAULT_TRANSPORT_TEMPERATURE_FLOOR,
) -> FaceTransportState:
    state = apply_transport_density_floor(state, density_floor)
    state = apply_transport_temperature_floor(state, temperature_floor, density_floor)
    density_faces = _face_profile(
        state.density,
        geometry.r_grid_half,
        bc_model=bc_density,
        reconstruction=reconstruction,
    )
    density_faces = safe_density(density_faces, density_floor)
    temperature_faces = _face_profile(
        state.temperature,
        geometry.r_grid_half,
        bc_model=bc_temperature,
        reconstruction=reconstruction,
    )
    temperature_faces = safe_temperature(temperature_faces, temperature_floor)
    pressure_faces = density_faces * temperature_faces
    er_faces = _face_profile(
        state.Er,
        geometry.r_grid_half,
        bc_model=bc_er,
        reconstruction=reconstruction,
    )
    return FaceTransportState(
        density=density_faces,
        pressure=pressure_faces,
        Er=er_faces,
    )


def _ntss_like_face_profile(profile, face_centers, bc_model=None, density_floor=None):
    if profile.ndim == 1:
        profile_2d = profile[None, :]
        squeeze = True
    else:
        profile_2d = profile
        squeeze = False
    left_value, _left_grad, right_value, _right_grad = _extract_face_constraints(
        bc_model,
        profile_2d,
        face_centers,
    )
    if left_value is None:
        left_value = profile_2d[:, 0]
    if right_value is None:
        right_value = profile_2d[:, -1]
    cell_centers = 0.5 * (face_centers[1:] + face_centers[:-1])
    inner = 0.5 * (profile_2d[:, :-1] + profile_2d[:, 1:])
    faces = jnp.concatenate([left_value[..., None], inner, right_value[..., None]], axis=-1)
    if density_floor is not None:
        faces = safe_density(faces, density_floor)
    if squeeze:
        return faces[0]
    return faces


def _ntss_like_face_gradient(profile, face_centers, bc_model=None):
    if profile.ndim == 1:
        profile_2d = profile[None, :]
        squeeze = True
    else:
        profile_2d = profile
        squeeze = False

    reference = _face_profile_gradient(profile_2d, face_centers, bc_model=bc_model)
    cell_centers = 0.5 * (face_centers[1:] + face_centers[:-1])
    inner = (profile_2d[:, 1:] - profile_2d[:, :-1]) / (cell_centers[1:] - cell_centers[:-1])
    grads = reference.at[:, 1:-1].set(inner)
    if squeeze:
        return grads[0]
    return grads


def build_ntss_like_face_transport_state(
    state: TransportState,
    geometry: Any,
    *,
    bc_density: Any = None,
    bc_temperature: Any = None,
    bc_er: Any = None,
    density_floor: Any = DEFAULT_TRANSPORT_DENSITY_FLOOR,
    temperature_floor: Any = DEFAULT_TRANSPORT_TEMPERATURE_FLOOR,
) -> FaceTransportState:
    state = apply_transport_density_floor(state, density_floor)
    state = apply_transport_temperature_floor(state, temperature_floor, density_floor)
    density_faces = _ntss_like_face_profile(
        state.density,
        geometry.r_grid_half,
        bc_model=bc_density,
        density_floor=density_floor,
    )
    temperature_faces = _ntss_like_face_profile(
        state.temperature,
        geometry.r_grid_half,
        bc_model=bc_temperature,
    )
    temperature_faces = safe_temperature(temperature_faces, temperature_floor)
    pressure_faces = density_faces * temperature_faces
    er_faces = _ntss_like_face_profile(
        state.Er,
        geometry.r_grid_half,
        bc_model=bc_er,
    )
    return FaceTransportState(
        density=density_faces,
        pressure=pressure_faces,
        Er=er_faces,
    )



@dataclasses.dataclass(frozen=True, eq=False)
class CombinedTransportFluxModel(TransportFluxModelBase):
    neoclassical_model: TransportFluxModelBase
    turbulent_model: TransportFluxModelBase
    classical_model: TransportFluxModelBase

    def __call__(self, state, *args, **kwargs) -> dict:
        # Only pass 'state' to the model instances, as expected by their __call__
        neo = self.neoclassical_model(state)
        turb = self.turbulent_model(state)
        classical = self.classical_model(state)
        out = {
            "Gamma": neo.get("Gamma", 0) + turb.get("Gamma", 0) + classical.get("Gamma", 0),
            "Q":     neo.get("Q", 0)     + turb.get("Q", 0)     + classical.get("Q", 0),
            "Upar":  neo.get("Upar", 0)  + turb.get("Upar", 0)  + classical.get("Upar", 0),
            "Gamma_neo": neo.get("Gamma", 0),
            "Q_neo":     neo.get("Q", 0),
            "Upar_neo":  neo.get("Upar", 0),
            "Gamma_turb": turb.get("Gamma", 0),
            "Q_turb":     turb.get("Q", 0),
            "Upar_turb":  turb.get("Upar", 0),
            "Gamma_classical": classical.get("Gamma", 0),
            "Q_classical":     classical.get("Q", 0),
            "Upar_classical":  classical.get("Upar", 0),
        }
        return out

    def build_local_particle_flux_evaluator(self, state):
        neo_eval = self.neoclassical_model.build_local_particle_flux_evaluator(state)
        turb_eval = self.turbulent_model.build_local_particle_flux_evaluator(state)
        classical_eval = self.classical_model.build_local_particle_flux_evaluator(state)
        if neo_eval is None or turb_eval is None or classical_eval is None:
            return None

        def evaluator(radius_index, er_value):
            return neo_eval(radius_index, er_value) + turb_eval(radius_index, er_value) + classical_eval(radius_index, er_value)

        return evaluator

    def evaluate_face_fluxes(self, state, face_state, **kwargs):
        neo = self.neoclassical_model.evaluate_face_fluxes(state, face_state, **kwargs)
        turb = self.turbulent_model.evaluate_face_fluxes(state, face_state, **kwargs)
        classical = self.classical_model.evaluate_face_fluxes(state, face_state, **kwargs)
        if neo is None or turb is None or classical is None:
            return None
        return {
            "Gamma": neo.get("Gamma", 0) + turb.get("Gamma", 0) + classical.get("Gamma", 0),
            "Q": neo.get("Q", 0) + turb.get("Q", 0) + classical.get("Q", 0),
            "Upar": neo.get("Upar", 0) + turb.get("Upar", 0) + classical.get("Upar", 0),
            "Gamma_neo": neo.get("Gamma", 0),
            "Q_neo": neo.get("Q", 0),
            "Upar_neo": neo.get("Upar", 0),
            "Gamma_turb": turb.get("Gamma", 0),
            "Q_turb": turb.get("Q", 0),
            "Upar_turb": turb.get("Upar", 0),
            "Gamma_classical": classical.get("Gamma", 0),
            "Q_classical": classical.get("Q", 0),
            "Upar_classical": classical.get("Upar", 0),
        }




@dataclasses.dataclass(frozen=True, eq=False)
class MonkesDatabaseTransportModel(TransportFluxModelBase):
    species: Any
    energy_grid: Any
    geometry: Any
    database: Any
    collisionality_model: str = "default"
    bc_density: Any = None
    bc_temperature: Any = None

    def __call__(self, state) -> dict:
        density = safe_density(state.density)
        _, gamma_neo, q_neo, upar_neo = get_Neoclassical_Fluxes(
            self.species,
            self.energy_grid,
            self.geometry,
            self.database,
            state.Er,
            state.temperature,
            density,
            collisionality_model=self.collisionality_model,
        )
        return {
            "Gamma": gamma_neo,
            "Q": q_neo,
            "Upar": upar_neo,
        }

    def build_local_particle_flux_evaluator(self, state):
        species = self.species
        energy_grid = self.energy_grid
        geometry = self.geometry
        database = self.database
        density = safe_density(state.density)
        temperature = state.temperature
        density_right_constraint, density_right_grad_constraint = _extract_right_constraints(
            self.bc_density,
            density,
        )
        temperature_right_constraint, temperature_right_grad_constraint = _extract_right_constraints(
            self.bc_temperature,
            temperature,
        )

        def evaluator(radius_index, er_value):
            er_scalar = jnp.asarray(er_value, dtype=state.Er.dtype)
            er_profile = state.Er.at[radius_index].set(er_scalar)
            _, gamma_neo, _, _ = get_Neoclassical_Fluxes(
                species,
                energy_grid,
                geometry,
                database,
                er_profile,
                temperature,
                density,
                density_right_constraint=density_right_constraint,
                density_right_grad_constraint=density_right_grad_constraint,
                temperature_right_constraint=temperature_right_constraint,
                temperature_right_grad_constraint=temperature_right_grad_constraint,
                collisionality_model=self.collisionality_model,
            )
            return gamma_neo[:, radius_index]

        return evaluator

    def evaluate_face_fluxes(self, state, face_state, **kwargs):
        density = safe_density(state.density)
        face_density = safe_density(face_state.density)
        bc_density = kwargs.get("bc_density")
        bc_temperature = kwargs.get("bc_temperature")
        particle_face_closure_mode = str(kwargs.get("particle_face_closure_mode", "reconstructed")).strip().lower()
        if particle_face_closure_mode in {"ntss_like", "ntss", "half_point"}:
            dndr_faces = _ntss_like_face_gradient(
                density,
                self.geometry.r_grid_half,
                bc_model=bc_density,
            )
            dTdr_faces = _ntss_like_face_gradient(
                state.temperature,
                self.geometry.r_grid_half,
                bc_model=bc_temperature,
            )
        else:
            dndr_faces = _face_profile_gradient(
                density,
                self.geometry.r_grid_half,
                bc_model=bc_density,
            )
            dTdr_faces = _face_profile_gradient(
                state.temperature,
                self.geometry.r_grid_half,
                bc_model=bc_temperature,
            )
        _, gamma_neo, q_neo, upar_neo = get_Neoclassical_Fluxes_Faces(
            self.species,
            self.energy_grid,
            self.geometry,
            self.database,
            face_state.Er,
            face_state.temperature,
            face_density,
            dndr_faces,
            dTdr_faces,
            collisionality_model=self.collisionality_model,
        )
        return {
            "Gamma": gamma_neo,
            "Q": q_neo,
            "Upar": upar_neo,
        }
    


# --- Torax-style, JAX-friendly ZeroTransportModel ---
@dataclasses.dataclass(frozen=True, eq=False)
class ZeroTransportModel(TransportFluxModelBase):
    shape: Any = None

    def __call__(self, state) -> dict:
        arr_shape = self.shape if self.shape is not None else state.density.shape
        gamma = jnp.zeros(arr_shape)
        q = jnp.zeros(arr_shape)
        upar = jnp.zeros(arr_shape)
        return {"Gamma": gamma, "Q": q, "Upar": upar}

    def build_local_particle_flux_evaluator(self, state):
        zeros = jnp.zeros(state.density.shape[0], dtype=state.density.dtype)

        def evaluator(radius_index, er_value):
            del radius_index, er_value
            return zeros

        return evaluator

    def evaluate_face_fluxes(self, state, face_state, **kwargs):
        del state, kwargs
        arr_shape = self.shape if self.shape is not None else face_state.density.shape
        gamma = jnp.zeros(arr_shape)
        q = jnp.zeros(arr_shape)
        upar = jnp.zeros(arr_shape)
        return {"Gamma": gamma, "Q": q, "Upar": upar}


def _normalize_flux_dataset(arr, n_species):
    out = jnp.asarray(arr, dtype=float)
    if out.ndim == 1:
        out = out[None, :]
    elif out.ndim != 2:
        raise ValueError(f"Flux dataset must be 1D or 2D, got shape {out.shape}.")

    if out.shape[0] == n_species:
        return out
    if out.shape[1] == n_species:
        return jnp.swapaxes(out, 0, 1)
    if out.shape[0] == 1:
        return jnp.repeat(out, n_species, axis=0)
    if out.shape[1] == 1:
        return jnp.repeat(jnp.swapaxes(out, 0, 1), n_species, axis=0)
    raise ValueError(f"Flux dataset shape {out.shape} is not compatible with n_species={n_species}.")


def read_flux_profile_file(path, n_species):
    with h5py.File(path, "r") as f:
        keys = set(f.keys())

        def _first(*names):
            for name in names:
                if name in keys:
                    return f[name][...]
            return None

        r = _first("r", "rho", "r_grid", "radius")
        if r is None:
            raise ValueError(f"Flux file '{path}' must contain one of datasets: r, rho, r_grid, radius.")
        gamma = _first("Gamma", "gamma")
        q = _first("Q", "q")
        upar = _first("Upar", "upar", "u_par")

    if gamma is None and q is None and upar is None:
        raise ValueError(f"Flux file '{path}' must contain at least one of Gamma, Q, or Upar.")

    r_arr = jnp.ravel(jnp.asarray(r, dtype=float))
    gamma_arr = None if gamma is None else _normalize_flux_dataset(gamma, n_species)
    q_arr = None if q is None else _normalize_flux_dataset(q, n_species)
    upar_arr = None if upar is None else _normalize_flux_dataset(upar, n_species)
    return r_arr, gamma_arr, q_arr, upar_arr


def build_fluxes_r_file_transport_model(
    species,
    geometry,
    *,
    fluxes_file=None,
    file=None,
    flux_file=None,
    neoclassical_file=None,
    turbulence_file=None,
    classical_file=None,
    **kwargs,
):
    del kwargs
    path = (
        fluxes_file
        or file
        or flux_file
        or neoclassical_file
        or turbulence_file
        or classical_file
    )
    if path is None:
        raise ValueError(
            "fluxes_r_file requires a flux file. "
            "Provide one of: fluxes_file, file, flux_file, neoclassical_file, turbulence_file, or classical_file."
        )
    r_data, gamma_data, q_data, upar_data = read_flux_profile_file(path, species.number_species)
    return FluxesRFileTransportModel(
        species=species,
        geometry=geometry,
        r_data=r_data,
        gamma_data=gamma_data,
        q_data=q_data,
        upar_data=upar_data,
    )


@dataclasses.dataclass(frozen=True, eq=False)
class FluxesRFileTransportModel(TransportFluxModelBase):
    species: Any
    geometry: Any
    r_data: Any
    gamma_data: Any = None
    q_data: Any = None
    upar_data: Any = None

    def _interp_species_profile(self, data, target_r):
        if data is None:
            return jnp.zeros((self.species.number_species, target_r.shape[0]), dtype=target_r.dtype)
        return jax.vmap(lambda prof: interpax.interp1d(self.r_data, prof, target_r))(data)

    def __call__(self, state) -> dict:
        del state
        gamma = self._interp_species_profile(self.gamma_data, self.geometry.r_grid)
        q = self._interp_species_profile(self.q_data, self.geometry.r_grid)
        upar = self._interp_species_profile(self.upar_data, self.geometry.r_grid)
        return {"Gamma": gamma, "Q": q, "Upar": upar}

    def build_local_particle_flux_evaluator(self, state):
        del state
        gamma = self._interp_species_profile(self.gamma_data, self.geometry.r_grid)

        def evaluator(radius_index, er_value):
            del er_value
            return gamma[:, radius_index]

        return evaluator

    def evaluate_face_fluxes(self, state, face_state, **kwargs):
        del state, face_state, kwargs
        gamma = self._interp_species_profile(self.gamma_data, self.geometry.r_grid_half)
        q = self._interp_species_profile(self.q_data, self.geometry.r_grid_half)
        upar = self._interp_species_profile(self.upar_data, self.geometry.r_grid_half)
        return {"Gamma": gamma, "Q": q, "Upar": upar}






# --- Torax-style, JAX-friendly AnalyticalTurbulentTransportModel ---
@dataclasses.dataclass(frozen=True, eq=False)
class AnalyticalTurbulentTransportModel(TransportFluxModelBase):
    species: Any
    grid: Any
    chi_t: Any
    chi_n: Any
    field: Any

    def __call__(self, state) -> dict:
        gamma_turb, q_turb = get_Turbulent_Fluxes_Analytical(
            self.species,
            self.grid,
            self.chi_t,
            self.chi_n,
            state.temperature,
            state.density,
            self.field,
        )
        upar = jnp.zeros_like(state.density)
        return {"Gamma": gamma_turb, "Q": q_turb, "Upar": upar}

    def build_local_particle_flux_evaluator(self, state):
        gamma_turb, _ = get_Turbulent_Fluxes_Analytical(
            self.species,
            self.grid,
            self.chi_t,
            self.chi_n,
            state.temperature,
            state.density,
            self.field,
        )

        def evaluator(radius_index, er_value):
            del er_value
            return gamma_turb[:, radius_index]

        return evaluator

    def evaluate_face_fluxes(self, state, face_state, **kwargs):
        bc_density = kwargs.get("bc_density")
        bc_temperature = kwargs.get("bc_temperature")
        dndr_faces = _face_profile_gradient(
            DENSITY_STATE_TO_PHYSICAL * state.density,
            self.field.r_grid_half,
            bc_model=bc_density,
        )
        dTdr_faces = _face_profile_gradient(
            TEMPERATURE_STATE_TO_PHYSICAL * state.temperature,
            self.field.r_grid_half,
            bc_model=bc_temperature,
        )
        gamma = -self.chi_n[:, None] * dndr_faces
        q = -(DENSITY_STATE_TO_PHYSICAL * face_state.density) * self.chi_t[:, None] * dTdr_faces
        upar = jnp.zeros_like(gamma)
        return {"Gamma": gamma, "Q": q, "Upar": upar}


@dataclasses.dataclass(frozen=True, eq=False)
class PowerAnalyticalTurbulentTransportModel(TransportFluxModelBase):
    species: Any
    field: Any
    chi_t: Any
    chi_n: Any
    pressure_source_model: Any = None
    total_power_mw: Any = None

    def _effective_total_power_mw(self, state):
        if self.total_power_mw is not None:
            return jnp.asarray(self.total_power_mw, dtype=state.density.dtype)
        return compute_total_power_mw(
            state,
            self.species,
            self.pressure_source_model,
            self.field,
        )

    def __call__(self, state) -> dict:
        total_power_mw = self._effective_total_power_mw(state)
        gamma_turb, q_turb = get_Turbulent_Fluxes_PowerOverN(
            self.species,
            self.chi_t,
            self.chi_n,
            total_power_mw,
            state.temperature,
            state.density,
            self.field,
        )
        upar = jnp.zeros_like(state.density)
        return {"Gamma": gamma_turb, "Q": q_turb, "Upar": upar}

    def build_local_particle_flux_evaluator(self, state):
        total_power_mw = self._effective_total_power_mw(state)
        gamma_turb, _ = get_Turbulent_Fluxes_PowerOverN(
            self.species,
            self.chi_t,
            self.chi_n,
            total_power_mw,
            state.temperature,
            state.density,
            self.field,
        )

        def evaluator(radius_index, er_value):
            del er_value
            return gamma_turb[:, radius_index]

        return evaluator

    def evaluate_face_fluxes(self, state, face_state, **kwargs):
        bc_density = kwargs.get("bc_density")
        bc_temperature = kwargs.get("bc_temperature")
        total_power_mw = self._effective_total_power_mw(state)
        dndr_faces = _face_profile_gradient(
            DENSITY_STATE_TO_PHYSICAL * state.density,
            self.field.r_grid_half,
            bc_model=bc_density,
        )
        dTdr_faces = _face_profile_gradient(
            TEMPERATURE_STATE_TO_PHYSICAL * state.temperature,
            self.field.r_grid_half,
            bc_model=bc_temperature,
        )
        electron_idx = int(self.species.species_idx["e"])
        ne_face = jnp.maximum(jnp.asarray(face_state.density[electron_idx], dtype=state.density.dtype), 1.0e-12)
        p075 = jnp.where(total_power_mw < 0.0, jnp.asarray(3.0, dtype=state.density.dtype), jnp.power(total_power_mw, 0.75))
        density_coeff = jnp.asarray(self.chi_n, dtype=state.density.dtype)[:, None] * p075 / ne_face[None, :]
        heat_coeff = jnp.asarray(self.chi_t, dtype=state.density.dtype)[:, None] * p075 / ne_face[None, :]
        gamma = -density_coeff * dndr_faces
        q = -(DENSITY_STATE_TO_PHYSICAL * face_state.density) * heat_coeff * dTdr_faces
        upar = jnp.zeros_like(gamma)
        return {"Gamma": gamma, "Q": q, "Upar": upar}


# --- PATCH: Accept [neoclassical]/flux_model and [turbulence]/model as defaults ---

# --- Refactored: Only the orchestrator builds models; this function is now a pure factory ---
def build_transport_flux_model(neo_model: TransportFluxModelBase,
                              turb_model: TransportFluxModelBase,
                              classical_model: TransportFluxModelBase = None) -> CombinedTransportFluxModel:
    """
    Build the composed transport model from explicit model instances.
    All models must be constructed up front by the orchestrator.
    """
    if classical_model is None:
        classical_model = ZeroTransportModel()
    return CombinedTransportFluxModel(neo_model, turb_model, classical_model)

register_transport_flux_model(
    "monkes_database",
    lambda species, energy_grid, geometry, database, collisionality_model="default", bc_density=None, bc_temperature=None: MonkesDatabaseTransportModel(
        species=species,
        energy_grid=energy_grid,
        geometry=geometry,
        database=database,
        collisionality_model=collisionality_model,
        bc_density=bc_density,
        bc_temperature=bc_temperature,
    ),
)


register_transport_flux_model(
    "monkes_database_with_momentum",
    lambda species, energy_grid, geometry, database,
           density_right_constraint=None, density_right_grad_constraint=None,
           temperature_right_constraint=None, temperature_right_grad_constraint=None: MonkesDatabaseWithMomentumTransportModel(
        species=species,
        energy_grid=energy_grid,
        geometry=geometry,
        database=database,
        density_right_constraint=density_right_constraint,
        density_right_grad_constraint=density_right_grad_constraint,
        temperature_right_constraint=temperature_right_constraint,
        temperature_right_grad_constraint=temperature_right_grad_constraint,
    ),
)

register_transport_flux_model(
    "turbulent_analytical",
    lambda species, grid, chi_t, chi_n, field: AnalyticalTurbulentTransportModel(
        species=species,
        grid=grid,
        chi_t=chi_t,
        chi_n=chi_n,
        field=field,
    ),
)

register_transport_flux_model(
    "turbulent_power_analytical",
    lambda species, grid, field, chi_t, chi_n, pressure_source_model=None, total_power_mw=None: PowerAnalyticalTurbulentTransportModel(
        species=species,
        field=field,
        chi_t=chi_t,
        chi_n=chi_n,
        pressure_source_model=pressure_source_model,
        total_power_mw=total_power_mw,
    ),
)

register_transport_flux_model(
    "ntss_power_over_n",
    lambda species, grid, field, chi_t, chi_n, pressure_source_model=None, total_power_mw=None: PowerAnalyticalTurbulentTransportModel(
        species=species,
        field=field,
        chi_t=chi_t,
        chi_n=chi_n,
        pressure_source_model=pressure_source_model,
        total_power_mw=total_power_mw,
    ),
)

register_transport_flux_model(
    "fluxes_r_file",
    lambda species, energy_grid, geometry, database, **kwargs: build_fluxes_r_file_transport_model(
        species=species,
        geometry=geometry,
        **kwargs,
    ),
)

register_transport_flux_model(
    "none",
    lambda *args, **kwargs: ZeroTransportModel(),
)
