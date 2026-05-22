
# transport_solvers.py
"""
Modular solver interface and backends for NEOPAX transport equations.
Supports multiple solver backends: time integration, root-finding, and optimization.
Inspired by torax (https://github.com/google-deepmind/torax).
"""

from typing import Callable, Any
import dataclasses
from functools import partial
import inspect
import time
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import roots_jacobi


TIME_SOLVER_REGISTRY: dict[str, Callable[..., "TransportSolver"]] = {}

ODE_SOLVER_BACKENDS = {
    "diffrax_kvaerno5",
    "diffrax_tsit5",
    "diffrax_dopri5",
    "radau",
    "theta",
    "theta_newton",
}


def _build_real_block_transform(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Return real-valued stage transform data for an odd-stage Radau tableau."""
    eigvals, eigvecs = np.linalg.eig(a)
    real_idx = int(np.argmin(np.abs(eigvals.imag)))
    real_vec = eigvecs[:, real_idx].real
    complex_indices = [idx for idx, eig in enumerate(eigvals) if eig.imag > 1.0e-12]
    complex_indices.sort(key=lambda idx: eigvals[idx].imag)
    columns = [real_vec]
    complex_blocks = []
    for idx in complex_indices:
        complex_vec = eigvecs[:, idx]
        columns.extend([complex_vec.real, complex_vec.imag])
        complex_blocks.append(
            [
                [float(eigvals[idx].real), float(eigvals[idx].imag)],
                [-float(eigvals[idx].imag), float(eigvals[idx].real)],
            ]
        )
    transform = np.column_stack(columns).astype(np.float64)
    inv_transform = np.linalg.inv(transform)
    real_eig = float(eigvals[real_idx].real)
    return transform, inv_transform, real_eig, np.asarray(complex_blocks, dtype=np.float64)

@dataclasses.dataclass(frozen=True)
class _RadauStageConfig:
    num_stages: int
    order: int
    embedded_order: int
    c: np.ndarray
    a: np.ndarray
    b: np.ndarray
    b_error: np.ndarray
    embedded_f0_weight: float
    has_embedded_estimator: bool
    transform: np.ndarray
    inv_transform: np.ndarray
    real_eig: float
    complex_blocks: np.ndarray


def _build_radau_iia_stage_config(num_stages: int) -> _RadauStageConfig:
    """Build fixed-stage Radau IIA coefficients and transformed solve data."""
    roots, _ = roots_jacobi(num_stages - 1, 1, 0)
    c = np.concatenate([np.sort((roots + 1.0) / 2.0), [1.0]]).astype(np.float64)
    polys: list[np.ndarray] = []
    for j in range(num_stages):
        poly = np.array([1.0], dtype=np.float64)
        denom = 1.0
        for m in range(num_stages):
            if m == j:
                continue
            poly = np.convolve(poly, np.array([-c[m], 1.0], dtype=np.float64))
            denom *= c[j] - c[m]
        polys.append(poly / denom)
    a = np.zeros((num_stages, num_stages), dtype=np.float64)
    b = np.zeros((num_stages,), dtype=np.float64)
    for j, poly in enumerate(polys):
        pint = np.zeros((len(poly) + 1,), dtype=np.float64)
        pint[1:] = poly / np.arange(1, len(poly) + 1, dtype=np.float64)
        for i, c_i in enumerate(c):
            a[i, j] = np.polyval(pint[::-1], c_i)
        b[j] = np.polyval(pint[::-1], 1.0)

    transform, inv_transform, real_eig, complex_blocks = _build_real_block_transform(a)
    if num_stages == 3:
        sqrt6 = np.sqrt(6.0)
        b_embedded = np.asarray([0.5 - 0.5 / sqrt6, 0.5 + 0.5 / sqrt6, 0.0], dtype=np.float64)
        b_error = b - b_embedded
        embedded_f0_weight = 0.0
        embedded_order = 2
        has_embedded = True
    else:
        vand = np.vstack([c ** q for q in range(num_stages)])
        rhs = np.asarray([1.0 - real_eig] + [1.0 / (q + 1) for q in range(1, num_stages)], dtype=np.float64)
        b_embedded = np.linalg.solve(vand, rhs)
        b_error = b - b_embedded
        embedded_f0_weight = -real_eig
        embedded_order = num_stages
        has_embedded = True

    return _RadauStageConfig(
        num_stages=num_stages,
        order=2 * num_stages - 1,
        embedded_order=embedded_order,
        c=c,
        a=a,
        b=b,
        b_error=b_error,
        embedded_f0_weight=embedded_f0_weight,
        has_embedded_estimator=has_embedded,
        transform=transform,
        inv_transform=inv_transform,
        real_eig=real_eig,
        complex_blocks=complex_blocks,
    )


_RADAU_STAGE_CONFIGS = {
    3: _build_radau_iia_stage_config(3),
    5: _build_radau_iia_stage_config(5),
    7: _build_radau_iia_stage_config(7),
    9: _build_radau_iia_stage_config(9),
    11: _build_radau_iia_stage_config(11),
}

def register_time_solver(name: str, builder: Callable[..., "TransportSolver"]) -> None:
    TIME_SOLVER_REGISTRY[str(name).strip().lower()] = builder


def _expects_time_argument(vector_field: Callable) -> bool:
    """Return True when vector_field signature starts with a time argument."""
    try:
        params = list(inspect.signature(vector_field).parameters.values())
    except (TypeError, ValueError):
        return False
    if len(params) < 2:
        return False
    return params[0].name in ("t", "time", "_t")


def _as_state_residual(vector_field: Callable) -> Callable:
    """Adapt a (t, y, ...) vector field to a pure state residual f(y, ...)."""
    if not _expects_time_argument(vector_field):
        return vector_field

    def residual(y, *args, **kwargs):
        return vector_field(jnp.asarray(0.0), y, *args, **kwargs)

    return residual


def _select_solver_family_and_backend(solver_parameters: Any) -> tuple[str, str]:
    """Pick the active Diffrax backend while preserving legacy integrator-only configs."""
    def _cfg_get(key: str, default=None):
        if isinstance(solver_parameters, dict):
            return solver_parameters.get(key, default)
        return getattr(solver_parameters, key, default)

    backend = str(
        _cfg_get(
            "transport_solver_backend",
            _cfg_get("integrator", "diffrax_kvaerno5"),
        )
    ).strip().lower()
    if backend not in ODE_SOLVER_BACKENDS:
        raise ValueError(
            f"Unsupported transport solver backend '{backend}'. "
            f"Expected one of {sorted(ODE_SOLVER_BACKENDS)}."
        )
    return "ode", backend


def _extract_species_from_args(args: tuple[Any, ...]) -> Any:
    if len(args) == 0:
        return None
    first = args[0]
    if isinstance(first, dict):
        return first.get("species", None)
    if hasattr(first, "charge_qp") and hasattr(first, "names"):
        return first
    if hasattr(first, "species"):
        return getattr(first, "species")
    return None


def _save_state_series(state: Any) -> Any:
    return jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), state)


def _save_scalar_series(value: Any) -> jax.Array:
    return jnp.expand_dims(jnp.asarray(value), axis=0)


def _strip_state_metadata_for_solver(state: Any) -> Any:
    """Hook for state cleanup before solver calls."""
    return state


def _electron_density_index(species: Any) -> int | None:
    if species is None or not hasattr(species, "names"):
        return None
    names = tuple(getattr(species, "names", ()))
    return names.index("e") if "e" in names else None


def _extract_fixed_temperature_projection(vector_field: Callable) -> tuple[Any, Any]:
    owner = getattr(vector_field, "__self__", None)
    if owner is None:
        return None, None
    return (
        getattr(owner, "temperature_active_mask", None),
        getattr(owner, "fixed_temperature_profile", None),
    )


def _extract_state_regularization(vector_field: Callable) -> tuple[Any, Any]:
    owner = getattr(vector_field, "__self__", None)
    if owner is None:
        return None, None
    return (
        getattr(owner, "density_floor", None),
        getattr(owner, "temperature_floor", None),
    )


def _pack_transport_state_arrays(state: Any, species: Any = None) -> Any:
    """Convert a TransportState-like object into an array-only pytree."""
    if dataclasses.is_dataclass(state) and hasattr(state, "density") and hasattr(state, "pressure") and hasattr(state, "Er"):
        density = state.density
        eidx = _electron_density_index(species)
        if eidx is not None:
            density = jnp.concatenate([density[:eidx], density[eidx + 1 :]], axis=0)
        return (density, state.pressure, state.Er)
    return state


def _unpack_transport_state_arrays(
    state_like: Any,
    template_state: Any,
    species: Any = None,
    temperature_active_mask: Any = None,
    fixed_temperature_profile: Any = None,
    density_floor: Any = None,
    temperature_floor: Any = None,
) -> Any:
    """Rebuild a TransportState-like object from an array-only pytree."""
    if (
        dataclasses.is_dataclass(template_state)
        and hasattr(template_state, "density")
        and hasattr(template_state, "pressure")
        and hasattr(template_state, "Er")
        and isinstance(state_like, tuple)
        and len(state_like) == 3
    ):
        density, pressure, er = state_like
        eidx = _electron_density_index(species)
        if eidx is not None and density.shape[-2] == template_state.density.shape[0] - 1:
            full_shape = pressure.shape[:-2] + (template_state.density.shape[0], pressure.shape[-1])
            full_density = jnp.zeros(full_shape, dtype=density.dtype)
            full_density = full_density.at[..., :eidx, :].set(density[..., :eidx, :])
            full_density = full_density.at[..., eidx + 1 :, :].set(density[..., eidx:, :])
            rebuilt = dataclasses.replace(template_state, density=full_density, pressure=pressure, Er=er)
            return _apply_quasi_neutrality_output(
                rebuilt,
                species,
                template_state,
                temperature_active_mask=temperature_active_mask,
                fixed_temperature_profile=fixed_temperature_profile,
                density_floor=density_floor,
                temperature_floor=temperature_floor,
            )
        rebuilt = dataclasses.replace(template_state, density=density, pressure=pressure, Er=er)
        return _project_fixed_temperature_output(
            rebuilt,
            template_state,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
            density_floor=density_floor,
            temperature_floor=temperature_floor,
        )
    return state_like


def _restore_state_metadata(state_like: Any, template_state: Any) -> Any:
    """Hook for restoring state metadata after solver calls."""
    return state_like


def _project_fixed_temperature_output(
    state_like: Any,
    reference_state: Any,
    temperature_active_mask: Any = None,
    fixed_temperature_profile: Any = None,
    density_floor: Any = None,
    temperature_floor: Any = None,
) -> Any:
    from ._transport_equations import project_fixed_temperature_species
    from ._state import apply_transport_density_floor, apply_transport_temperature_floor

    density = getattr(state_like, "density", None)
    ref_density = getattr(reference_state, "density", None)
    if density is None or ref_density is None:
        return state_like

    def _regularize_one(s):
        s = apply_transport_density_floor(s, density_floor)
        s = project_fixed_temperature_species(
            s,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
            density_floor=density_floor,
        )
        s = apply_transport_temperature_floor(s, temperature_floor, density_floor)
        return s

    if density.ndim == ref_density.ndim + 1:
        out = jax.vmap(_regularize_one)(state_like)
        return _restore_state_metadata(out, reference_state)
    out = _regularize_one(state_like)
    return _restore_state_metadata(out, reference_state)


def _apply_quasi_neutrality_output(
    state_like: Any,
    species: Any,
    reference_state: Any,
    temperature_active_mask: Any = None,
    fixed_temperature_profile: Any = None,
    density_floor: Any = None,
    temperature_floor: Any = None,
) -> Any:
    """Apply quasi-neutrality to either a single state or a saved time-series of states."""
    from ._transport_equations import enforce_quasi_neutrality

    density = getattr(state_like, "density", None)
    ref_density = getattr(reference_state, "density", None)
    if density is None or ref_density is None:
        return state_like

    if density.ndim == ref_density.ndim + 1:
        out = jax.vmap(lambda s: enforce_quasi_neutrality(s, species))(state_like)
        return _project_fixed_temperature_output(
            out,
            reference_state,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
            density_floor=density_floor,
            temperature_floor=temperature_floor,
        )
    out = enforce_quasi_neutrality(state_like, species)
    return _project_fixed_temperature_output(
        out,
        reference_state,
        temperature_active_mask=temperature_active_mask,
        fixed_temperature_profile=fixed_temperature_profile,
        density_floor=density_floor,
        temperature_floor=temperature_floor,
    )


def _project_state_to_quasi_neutrality(
    state_like: Any,
    species: Any,
    temperature_active_mask: Any = None,
    fixed_temperature_profile: Any = None,
    density_floor: Any = None,
    temperature_floor: Any = None,
) -> Any:
    from ._transport_equations import enforce_quasi_neutrality, project_fixed_temperature_species
    from ._state import apply_transport_density_floor, apply_transport_temperature_floor

    def _regularize_one(s):
        s = apply_transport_density_floor(s, density_floor)
        s = project_fixed_temperature_species(
            s,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
            density_floor=density_floor,
        )
        s = apply_transport_temperature_floor(s, temperature_floor, density_floor)
        return s

    if species is None:
        return _regularize_one(state_like)
    density = getattr(state_like, "density", None)
    if density is None:
        return _regularize_one(state_like)
    projected = enforce_quasi_neutrality(state_like, species)
    return _regularize_one(projected)


def _project_packed_transport_state_arrays(
    state_like: Any,
    template_state: Any,
    species: Any,
    temperature_active_mask: Any = None,
    fixed_temperature_profile: Any = None,
    density_floor: Any = None,
    temperature_floor: Any = None,
) -> Any:
    """Project packed solver arrays without rebuilding a full TransportState."""
    from ._state import safe_density, safe_temperature

    if not (isinstance(state_like, tuple) and len(state_like) == 3):
        return state_like

    density, pressure, er = state_like
    packed_density = safe_density(density, density_floor)
    eidx = _electron_density_index(species)
    if eidx is not None and packed_density.shape[-2] == template_state.density.shape[0] - 1:
        full_shape = pressure.shape[:-2] + (template_state.density.shape[0], pressure.shape[-1])
        full_density = jnp.zeros(full_shape, dtype=pressure.dtype)
        full_density = full_density.at[..., :eidx, :].set(packed_density[..., :eidx, :])
        full_density = full_density.at[..., eidx + 1 :, :].set(packed_density[..., eidx:, :])
        charge_qp = jnp.asarray(species.charge_qp, dtype=pressure.dtype)
        ion_indices = jnp.asarray(getattr(species, "ion_indices", ()), dtype=int)
        if ion_indices.size > 0:
            Z_i = jnp.take(charge_qp, ion_indices, axis=0)
            n_i = jnp.take(full_density, ion_indices, axis=-2)
            n_e = -jnp.sum(Z_i[..., None] * n_i, axis=-2) / charge_qp[int(eidx)]
            full_density = full_density.at[..., int(eidx), :].set(n_e)
    else:
        full_density = packed_density

    floored_density = safe_density(full_density, density_floor)
    projected_pressure = pressure
    if temperature_active_mask is not None and fixed_temperature_profile is not None:
        active_mask = jnp.asarray(temperature_active_mask, dtype=bool)
        fixed_temperature = jnp.asarray(fixed_temperature_profile, dtype=pressure.dtype)
        active_mask = active_mask.reshape((1,) * (pressure.ndim - 2) + (active_mask.shape[0], 1))
        fixed_temperature = jnp.broadcast_to(fixed_temperature, pressure.shape)
        fixed_pressure = floored_density * fixed_temperature
        projected_pressure = jnp.where(active_mask, projected_pressure, fixed_pressure)
    if temperature_floor is not None:
        projected_temperature = safe_temperature(projected_pressure / floored_density, temperature_floor)
        projected_pressure = floored_density * projected_temperature
    return (packed_density, projected_pressure, er)


def _project_flat_state_if_needed(
    flat_y: jax.Array,
    project_flat: Callable[[jax.Array], jax.Array] | None,
) -> jax.Array:
    if project_flat is None:
        return flat_y
    return project_flat(flat_y)


def _make_solver_state_transform(
    template_state: Any,
    species: Any,
    temperature_active_mask: Any = None,
    fixed_temperature_profile: Any = None,
    density_floor: Any = None,
    temperature_floor: Any = None,
):
    packed_state = _pack_transport_state_arrays(template_state, species)
    flat_state0, unravel_packed = jax.flatten_util.ravel_pytree(packed_state)

    def unpack_flat(flat_y):
        return _unpack_transport_state_arrays(
            unravel_packed(flat_y),
            template_state,
            species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
            density_floor=density_floor,
            temperature_floor=temperature_floor,
        )

    def unpack_packed(packed_state_like):
        return _unpack_transport_state_arrays(
            packed_state_like,
            template_state,
            species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
            density_floor=density_floor,
            temperature_floor=temperature_floor,
        )

    def pack_state(state_like):
        flat_out, _ = jax.flatten_util.ravel_pytree(_pack_transport_state_arrays(state_like, species))
        return flat_out

    def project_flat(flat_y):
        projected_packed = _project_packed_transport_state_arrays(
            unravel_packed(flat_y),
            template_state,
            species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
            density_floor=density_floor,
            temperature_floor=temperature_floor,
        )
        flat_projected, _ = jax.flatten_util.ravel_pytree(projected_packed)
        return flat_projected

    return flat_state0, unpack_flat, unpack_packed, pack_state, project_flat


def _get_diffrax_integrator(name: str) -> Callable:
    import diffrax

    key = str(name).strip().lower()
    mapping = {
        "diffrax_kvaerno5": diffrax.Kvaerno5,
        "diffrax_tsit5": diffrax.Tsit5,
        "diffrax_dopri5": diffrax.Dopri5,
    }
    if key not in mapping:
        raise ValueError(f"Unknown diffrax integrator '{name}'.")
    return mapping[key]

@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class TransportSolver:
    """
    Base class for transport solvers.
    All solvers must implement the solve() method.
    """
    def solve(self, state, vector_field: Callable, *args, **kwargs) -> Any:
        raise NotImplementedError

@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class DiffraxSolver(TransportSolver):
    integrator: Callable
    t0: float
    t1: float
    dt: float
    integrator_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __init__(self, integrator, t0, t1, dt, save_n=None, **integrator_kwargs):
        object.__setattr__(self, 'integrator', integrator)
        object.__setattr__(self, 't0', t0)
        object.__setattr__(self, 't1', t1)
        object.__setattr__(self, 'dt', dt)
        object.__setattr__(self, 'integrator_kwargs', integrator_kwargs)
        object.__setattr__(self, 'save_n', save_n)

    def solve(self, state, vector_field: Callable, *args, **kwargs):
        # Keep Diffrax on a flat array state to shrink the traced solve graph.
        # The physics still goes through the same NEOPAX vector field via the
        # shared pack/unpack/projection helpers used by the custom solvers.
        import diffrax
        import equinox as eqx
        species = _extract_species_from_args(args)
        temperature_active_mask, fixed_temperature_profile = _extract_fixed_temperature_projection(vector_field)
        density_floor, temperature_floor = _extract_state_regularization(vector_field)
        state = _project_state_to_quasi_neutrality(
            state,
            species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
            density_floor=density_floor,
            temperature_floor=temperature_floor,
        )
        flat_state0, unpack_flat, _unpack_packed, _pack_state, project_flat = _make_solver_state_transform(
            state,
            species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
            density_floor=density_floor,
            temperature_floor=temperature_floor,
        )
        flat_rhs = _flat_rhs_factory(unpack_flat, vector_field, args, kwargs, project_flat=project_flat)

        def wrapped_vector_field(t, flat_y, _vf_args):
            return flat_rhs(t, flat_y)

        term = diffrax.ODETerm(wrapped_vector_field)
        solver = self.integrator()
        call_kwargs = dict(self.integrator_kwargs)
        call_kwargs.update(kwargs)
        call_kwargs.setdefault("throw", False)
        save_n = getattr(self, 'save_n', None)
        if save_n is not None and save_n > 1:
            call_kwargs.pop("saveat", None)
            ts = jnp.linspace(self.t0, self.t1, save_n)
            saveat = diffrax.SaveAt(ts=ts)
            sol = diffrax.diffeqsolve(
                term,
                solver,
                t0=self.t0,
                t1=self.t1,
                dt0=self.dt,
                y0=flat_state0,
                args=None,
                saveat=saveat,
                **call_kwargs
            )
        else:
            sol = diffrax.diffeqsolve(
                term,
                solver,
                t0=self.t0,
                t1=self.t1,
                dt0=self.dt,
                y0=flat_state0,
                args=None,
                **call_kwargs
            )

        def _unpack_saved_values(flat_values):
            if isinstance(flat_values, jax.Array) and flat_values.ndim == flat_state0.ndim + 1:
                return jax.vmap(unpack_flat)(flat_values)
            return unpack_flat(flat_values)

        # Rebuild saved/final states and reapply the output-side algebraic
        # projections so solver outputs keep the same public semantics.
        if species is not None:
            if hasattr(sol, 'ys'):
                ys_state = _unpack_saved_values(sol.ys)
                sol = eqx.tree_at(
                    lambda s: s.ys,
                    sol,
                    _apply_quasi_neutrality_output(
                        ys_state,
                        species,
                        state,
                        temperature_active_mask=temperature_active_mask,
                        fixed_temperature_profile=fixed_temperature_profile,
                        density_floor=density_floor,
                        temperature_floor=temperature_floor,
                    ),
                )
        else:
            if hasattr(sol, 'ys'):
                ys_state = _unpack_saved_values(sol.ys)
                sol = eqx.tree_at(
                    lambda s: s.ys,
                    sol,
                    _project_fixed_temperature_output(
                        ys_state,
                        state,
                        temperature_active_mask=temperature_active_mask,
                        fixed_temperature_profile=fixed_temperature_profile,
                        density_floor=density_floor,
                        temperature_floor=temperature_floor,
                    ),
                )
        return sol


def _finalize_custom_solver_output(
    ys_saved_flat,
    ts_saved,
    dts_saved,
    accepted_mask_saved,
    failed_mask_saved,
    fail_codes_saved,
    y_final_flat,
    t_final,
    done_f,
    failed_f,
    fail_code_f,
    n_steps_f,
    last_attempt_accepted,
    last_attempt_converged,
    last_attempt_err_norm,
    last_attempt_fail_code,
    last_attempt_diverged,
    last_attempt_nonfinite_stage_state,
    last_attempt_nonfinite_stage_residual,
    last_attempt_finite_f0,
    last_attempt_finite_z0,
    last_attempt_finite_initial_residual,
    last_attempt_newton_iter_count,
    last_attempt_final_residual_norm,
    last_attempt_final_delta_norm,
    last_attempt_theta_final,
    last_attempt_slow_contraction,
    last_attempt_residual_blowup,
    last_attempt_newton_nonfinite,
    last_attempt_lagged_reused,
    last_attempt_jacobian_reused,
    final_reuse_state,
    unpack_flat,
    reference_state,
    species,
    temperature_active_mask=None,
    fixed_temperature_profile=None,
    density_floor=None,
    temperature_floor=None,
):
    from ._transport_equations import enforce_quasi_neutrality

    ys_saved = jax.vmap(unpack_flat)(ys_saved_flat)
    if species is not None:
        ys_saved = jax.vmap(lambda s: enforce_quasi_neutrality(s, species))(ys_saved)
    ys_saved = _project_fixed_temperature_output(
        ys_saved,
        reference_state,
        temperature_active_mask=temperature_active_mask,
        fixed_temperature_profile=fixed_temperature_profile,
        density_floor=density_floor,
        temperature_floor=temperature_floor,
    )
    final_state = unpack_flat(y_final_flat)
    if species is not None:
        final_state = enforce_quasi_neutrality(final_state, species)
    final_state = _project_fixed_temperature_output(
        final_state,
        reference_state,
        temperature_active_mask=temperature_active_mask,
        fixed_temperature_profile=fixed_temperature_profile,
        density_floor=density_floor,
        temperature_floor=temperature_floor,
    )
    reuse_lagged_available = None if final_reuse_state is None else getattr(final_reuse_state, "lagged_response_available", None)
    reuse_lagged_valid = None if final_reuse_state is None else getattr(final_reuse_state, "lagged_response_valid", None)
    reuse_cache_valid = None if final_reuse_state is None else getattr(final_reuse_state, "cache_valid", None)
    reuse_cache_dt = None if final_reuse_state is None else getattr(final_reuse_state, "cache_dt", None)
    reuse_cache_age = None if final_reuse_state is None else getattr(final_reuse_state, "cache_age", None)
    reuse_freeze_attempt_linearization = None if final_reuse_state is None else getattr(final_reuse_state, "freeze_attempt_linearization", None)
    reuse_last_lagged_reused = None if final_reuse_state is None else getattr(final_reuse_state, "last_lagged_reused", None)
    reuse_last_jacobian_reused = None if final_reuse_state is None else getattr(final_reuse_state, "last_jacobian_reused", None)
    reuse_last_linearization_dt = None if final_reuse_state is None else getattr(final_reuse_state, "last_linearization_dt", None)
    return {
        "ys": ys_saved,
        "ts": ts_saved,
        "dts": dts_saved,
        "accepted_mask": accepted_mask_saved,
        "failed_mask": failed_mask_saved,
        "fail_codes": fail_codes_saved,
        "n_steps": n_steps_f,
        "done": done_f,
        "failed": failed_f,
        "fail_code": fail_code_f,
        "last_attempt_accepted": last_attempt_accepted,
        "last_attempt_converged": last_attempt_converged,
        "last_attempt_err_norm": last_attempt_err_norm,
        "last_attempt_fail_code": last_attempt_fail_code,
        "last_attempt_diverged": last_attempt_diverged,
        "last_attempt_nonfinite_stage_state": last_attempt_nonfinite_stage_state,
        "last_attempt_nonfinite_stage_residual": last_attempt_nonfinite_stage_residual,
        "last_attempt_finite_f0": last_attempt_finite_f0,
        "last_attempt_finite_z0": last_attempt_finite_z0,
        "last_attempt_finite_initial_residual": last_attempt_finite_initial_residual,
        "last_attempt_newton_iter_count": last_attempt_newton_iter_count,
        "last_attempt_final_residual_norm": last_attempt_final_residual_norm,
        "last_attempt_final_delta_norm": last_attempt_final_delta_norm,
        "last_attempt_theta_final": last_attempt_theta_final,
        "last_attempt_slow_contraction": last_attempt_slow_contraction,
        "last_attempt_residual_blowup": last_attempt_residual_blowup,
        "last_attempt_newton_nonfinite": last_attempt_newton_nonfinite,
        "last_attempt_lagged_reused": last_attempt_lagged_reused,
        "last_attempt_jacobian_reused": last_attempt_jacobian_reused,
        "final_reuse_state": final_reuse_state,
        "final_reuse_lagged_response_available": reuse_lagged_available,
        "final_reuse_lagged_response_valid": reuse_lagged_valid,
        "final_reuse_cache_valid": reuse_cache_valid,
        "final_reuse_cache_dt": reuse_cache_dt,
        "final_reuse_cache_age": reuse_cache_age,
        "final_reuse_freeze_attempt_linearization": reuse_freeze_attempt_linearization,
        "final_reuse_last_lagged_reused": reuse_last_lagged_reused,
        "final_reuse_last_jacobian_reused": reuse_last_jacobian_reused,
        "final_reuse_last_linearization_dt": reuse_last_linearization_dt,
        "final_state": final_state,
        "final_time": t_final,
    }


def _fill_saved_slots(
    save_idx,
    save_times,
    t_value,
    flat_y,
    dt_value,
    accepted,
    failed,
    fail_code,
    ys,
    ts,
    dts,
    accs,
    fails,
    codes,
):
    save_n = save_times.shape[0]

    def cond_fn(loop_state):
        save_i, *_ = loop_state
        return jnp.logical_and(save_i < save_n, t_value >= save_times[save_i])

    def body_fn(loop_state):
        save_i, ys_l, ts_l, dts_l, accs_l, fails_l, codes_l = loop_state
        ys_l = ys_l.at[save_i].set(flat_y)
        ts_l = ts_l.at[save_i].set(t_value)
        dts_l = dts_l.at[save_i].set(dt_value)
        accs_l = accs_l.at[save_i].set(accepted)
        fails_l = fails_l.at[save_i].set(failed)
        codes_l = codes_l.at[save_i].set(fail_code)
        return save_i + 1, ys_l, ts_l, dts_l, accs_l, fails_l, codes_l

    return jax.lax.while_loop(
        cond_fn,
        body_fn,
        (save_idx, ys, ts, dts, accs, fails, codes),
    )


def _flat_rhs_factory(unravel, vector_field, args, kwargs, project_flat=None):
    species = _extract_species_from_args(args)

    def _flat_rhs(t_value, flat_y):
        projected_flat_y = _project_flat_state_if_needed(
            flat_y,
            project_flat,
        )
        rhs_tree = vector_field(t_value, unravel(projected_flat_y), *args, **kwargs)
        rhs_flat, _ = jax.flatten_util.ravel_pytree(_pack_transport_state_arrays(rhs_tree, species))
        return rhs_flat

    return _flat_rhs


def _lagged_response_hooks(vector_field: Callable):
    owner = getattr(vector_field, "__self__", None)
    if owner is None:
        return None, None
    build_fn = getattr(owner, "build_lagged_response", None)
    eval_fn = getattr(owner, "evaluate_with_lagged_response", None)
    if callable(build_fn) and callable(eval_fn):
        return build_fn, eval_fn
    return None, None


def _flat_rhs_with_lagged_response_factory(unravel, vector_field, args, kwargs, project_flat=None):
    species = _extract_species_from_args(args)
    _, eval_fn = _lagged_response_hooks(vector_field)

    def _flat_rhs(t_value, flat_y, lagged_response):
        projected_flat_y = _project_flat_state_if_needed(
            flat_y,
            project_flat,
        )
        state_y = unravel(projected_flat_y)
        if eval_fn is not None:
            rhs_tree = eval_fn(t_value, state_y, *args, lagged_response=lagged_response, **kwargs)
        else:
            rhs_tree = vector_field(t_value, state_y, *args, **kwargs)
        rhs_flat, _ = jax.flatten_util.ravel_pytree(_pack_transport_state_arrays(rhs_tree, species))
        return rhs_flat

    return _flat_rhs


def _solver_error_norm(err_vec, flat_ref, flat_candidate, atol: float, rtol: float, scale_mode: str = "max", rtol_eff=None, scale_override=None):
    ref_abs = jnp.abs(flat_ref)
    cand_abs = jnp.abs(flat_candidate)
    max_scale = jnp.maximum(ref_abs, cand_abs)
    mean_scale = 0.5 * (ref_abs + cand_abs)
    effective_rtol = rtol if rtol_eff is None else rtol_eff
    if scale_override is None:
        if scale_mode == "max":
            scale_base = max_scale
        elif scale_mode == "mean":
            scale_base = mean_scale
        elif scale_mode == "blend":
            scale_base = jnp.asarray(0.75, dtype=err_vec.dtype) * max_scale + jnp.asarray(0.25, dtype=err_vec.dtype) * mean_scale
        elif scale_mode == "ntss":
            scale_base = cand_abs
        elif scale_mode == "ntss_max":
            scale_base = max_scale
        elif scale_mode == "ntss_blend":
            scale_base = jnp.asarray(0.5, dtype=err_vec.dtype) * cand_abs + jnp.asarray(0.5, dtype=err_vec.dtype) * max_scale
        elif scale_mode in {"ntss_transport", "ntss_block_floor", "ntss_block_rms"}:
            # These modes normally provide an explicit `scale_override` from the
            # caller. If that is unavailable, fall back to the same max-based
            # scaling used by the default embedded estimator path.
            scale_base = max_scale
        else:
            raise ValueError(f"Unsupported solver error norm scale_mode '{scale_mode}'.")
        scale = atol + effective_rtol * scale_base
    else:
        scale = scale_override
    normalized = err_vec / scale
    return jnp.sqrt(jnp.mean(normalized * normalized) + 1.0e-30)


def _lagged_response_global_reuse_metric(current_flat, reference_flat, atol: float, rtol: float):
    delta_flat = current_flat - reference_flat
    return _solver_error_norm(delta_flat, reference_flat, current_flat, atol=atol, rtol=rtol)


def _make_radau_initial_step_state(
    t0,
    flat_state0,
    base_dt,
    dtype,
    initial_rhs,
    num_stages,
    real_lu0,
    real_piv0,
    complex_lu0,
    complex_piv0,
    lagged_response_cache,
    lagged_response_valid,
    lagged_reference_y,
):
    state_dim = flat_state0.shape[0]
    carry0 = _RadauAcceptedStepCarry(
        t=t0,
        y=flat_state0,
        dt=base_dt,
        prev_error=jnp.asarray(1.0, dtype=dtype),
        prev_stages=jnp.tile(initial_rhs, num_stages),
        prev_dt=jnp.asarray(0.0, dtype=dtype),
        recent_reject_count=jnp.asarray(0, dtype=jnp.int32),
        regrowth_cooldown=jnp.asarray(0, dtype=jnp.int32),
        easy_growth_streak=jnp.asarray(0, dtype=jnp.int32),
        lagged_response_cache=lagged_response_cache,
        lagged_response_valid=lagged_response_valid,
        lagged_reference_y=lagged_reference_y,
        jacobian=jnp.zeros((state_dim, state_dim), dtype=dtype),
        cache_valid=jnp.asarray(False),
        cache_dt=jnp.asarray(0.0, dtype=dtype),
        cache_age=jnp.asarray(0, dtype=jnp.int32),
        real_lu=real_lu0,
        real_piv=real_piv0,
        complex_lu=complex_lu0,
        complex_piv=complex_piv0,
        prev_theta_final=jnp.asarray(0.0, dtype=dtype),
        prev_newton_iter_count=jnp.asarray(0, dtype=jnp.int32),
    )
    return _radau_step_state_from_carry(
        carry0,
        status=jnp.asarray([0, 0, 0], dtype=jnp.int32),
    )


def _custom_loop_active(step_state, t_final, step_idx, max_total_steps):
    failed = step_state.status[0] != 0
    active = jnp.logical_and(step_state.t < (t_final - 1.0e-15), jnp.logical_not(failed))
    return jnp.logical_and(step_idx < max_total_steps, active)


def _accepted_step_limit_reached(step_state, stop_after_accepted_steps):
    if stop_after_accepted_steps is None:
        return jnp.asarray(False)
    accepted_steps = step_state.status[2]
    accepted_limit = jnp.asarray(stop_after_accepted_steps, dtype=accepted_steps.dtype)
    return accepted_steps >= accepted_limit


def _run_saved_loop(
    *,
    step_state0,
    step_fn,
    save_n,
    t0,
    t_final,
    state_dim,
    dtype,
    max_total_steps,
    stop_after_accepted_steps=None,
):
    save_times = jnp.linspace(t0, t_final, save_n)
    ys_saved = jnp.zeros((save_n, state_dim), dtype=dtype)
    ts_saved = jnp.zeros((save_n,), dtype=dtype)
    dts_saved = jnp.zeros((save_n,), dtype=dtype)
    accepted_mask_saved = jnp.zeros((save_n,), dtype=bool)
    failed_mask_saved = jnp.zeros((save_n,), dtype=bool)
    fail_codes_saved = jnp.zeros((save_n,), dtype=jnp.int32)
    ys_saved = ys_saved.at[0].set(step_state0.y)
    ts_saved = ts_saved.at[0].set(t0)
    accepted_mask_saved = accepted_mask_saved.at[0].set(True)
    last_attempt_accepted0 = jnp.asarray(False)
    last_attempt_converged0 = jnp.asarray(False)
    last_attempt_err_norm0 = jnp.asarray(jnp.inf, dtype=dtype)
    last_attempt_fail_code0 = jnp.asarray(0, dtype=jnp.int32)
    last_attempt_diverged0 = jnp.asarray(False)
    last_attempt_nonfinite_stage_state0 = jnp.asarray(False)
    last_attempt_nonfinite_stage_residual0 = jnp.asarray(False)
    last_attempt_finite_f00 = jnp.asarray(True)
    last_attempt_finite_z00 = jnp.asarray(True)
    last_attempt_finite_initial_residual0 = jnp.asarray(True)
    last_attempt_newton_iter_count0 = jnp.asarray(0, dtype=jnp.int32)
    last_attempt_final_residual_norm0 = jnp.asarray(jnp.inf, dtype=dtype)
    last_attempt_final_delta_norm0 = jnp.asarray(jnp.inf, dtype=dtype)
    last_attempt_theta_final0 = jnp.asarray(0.0, dtype=dtype)
    last_attempt_slow_contraction0 = jnp.asarray(False)
    last_attempt_residual_blowup0 = jnp.asarray(False)
    last_attempt_newton_nonfinite0 = jnp.asarray(False)
    last_attempt_lagged_reused0 = jnp.asarray(False)
    last_attempt_jacobian_reused0 = jnp.asarray(False)

    def cond_fun(loop_carry):
        step_state, step_idx, *_ = loop_carry
        active = _custom_loop_active(step_state, t_final, step_idx, max_total_steps)
        return jnp.logical_and(active, jnp.logical_not(_accepted_step_limit_reached(step_state, stop_after_accepted_steps)))

    def body_fun(loop_carry):
        (
            step_state,
            step_idx,
            save_idx,
            ys,
            ts,
            dts,
            accs,
            fails,
            codes,
            _last_accepted,
            _last_converged,
            _last_err_norm,
            _last_fail_code,
            _last_diverged,
            _last_nonfinite_stage_state,
            _last_nonfinite_stage_residual,
            _last_finite_f0,
            _last_finite_z0,
            _last_finite_initial_residual,
            _last_newton_iter_count,
            _last_final_residual_norm,
            _last_final_delta_norm,
            _last_theta_final,
            _last_slow_contraction,
            _last_residual_blowup,
            _last_newton_nonfinite,
            _last_lagged_reused,
            _last_jacobian_reused,
        ) = loop_carry
        step_state, step_info = step_fn(step_state, None)
        save_idx, ys, ts, dts, accs, fails, codes = _fill_saved_slots(
            save_idx,
            save_times,
            step_info.t,
            step_info.y,
            step_info.dt,
            step_info.accepted,
            step_info.failed,
            step_info.fail_code,
            ys,
            ts,
            dts,
            accs,
            fails,
            codes,
        )
        return (
            step_state,
            step_idx + 1,
            save_idx,
            ys,
            ts,
            dts,
            accs,
            fails,
            codes,
            jnp.asarray(step_info.accepted),
            jnp.asarray(False if getattr(step_info, "converged", None) is None else getattr(step_info, "converged")),
            jnp.asarray(jnp.inf, dtype=dtype) if getattr(step_info, "err_norm", None) is None else jnp.asarray(getattr(step_info, "err_norm"), dtype=dtype),
            jnp.asarray(step_info.fail_code, dtype=jnp.int32),
            jnp.asarray(False if getattr(step_info, "diverged", None) is None else getattr(step_info, "diverged")),
            jnp.asarray(False if getattr(step_info, "nonfinite_stage_state", None) is None else getattr(step_info, "nonfinite_stage_state")),
            jnp.asarray(False if getattr(step_info, "nonfinite_stage_residual", None) is None else getattr(step_info, "nonfinite_stage_residual")),
            jnp.asarray(True if getattr(step_info, "finite_f0", None) is None else getattr(step_info, "finite_f0")),
            jnp.asarray(True if getattr(step_info, "finite_z0", None) is None else getattr(step_info, "finite_z0")),
            jnp.asarray(True if getattr(step_info, "finite_initial_residual", None) is None else getattr(step_info, "finite_initial_residual")),
            jnp.asarray(0 if getattr(step_info, "newton_iter_count", None) is None else getattr(step_info, "newton_iter_count"), dtype=jnp.int32),
            jnp.asarray(jnp.inf, dtype=dtype) if getattr(step_info, "final_residual_norm", None) is None else jnp.asarray(getattr(step_info, "final_residual_norm"), dtype=dtype),
            jnp.asarray(jnp.inf, dtype=dtype) if getattr(step_info, "final_delta_norm", None) is None else jnp.asarray(getattr(step_info, "final_delta_norm"), dtype=dtype),
            jnp.asarray(0.0, dtype=dtype) if getattr(step_info, "theta_final", None) is None else jnp.asarray(getattr(step_info, "theta_final"), dtype=dtype),
            jnp.asarray(False if getattr(step_info, "slow_contraction", None) is None else getattr(step_info, "slow_contraction")),
            jnp.asarray(False if getattr(step_info, "residual_blowup", None) is None else getattr(step_info, "residual_blowup")),
            jnp.asarray(False if getattr(step_info, "newton_nonfinite", None) is None else getattr(step_info, "newton_nonfinite")),
            jnp.asarray(False if getattr(step_info, "lagged_reused", None) is None else getattr(step_info, "lagged_reused")),
            jnp.asarray(False if getattr(step_info, "jacobian_reused", None) is None else getattr(step_info, "jacobian_reused")),
        )

    loop_carry = (
        step_state0,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(1, dtype=jnp.int32),
        ys_saved,
        ts_saved,
        dts_saved,
        accepted_mask_saved,
        failed_mask_saved,
        fail_codes_saved,
        last_attempt_accepted0,
        last_attempt_converged0,
        last_attempt_err_norm0,
        last_attempt_fail_code0,
        last_attempt_diverged0,
        last_attempt_nonfinite_stage_state0,
        last_attempt_nonfinite_stage_residual0,
        last_attempt_finite_f00,
        last_attempt_finite_z00,
        last_attempt_finite_initial_residual0,
        last_attempt_newton_iter_count0,
        last_attempt_final_residual_norm0,
        last_attempt_final_delta_norm0,
        last_attempt_theta_final0,
        last_attempt_slow_contraction0,
        last_attempt_residual_blowup0,
        last_attempt_newton_nonfinite0,
        last_attempt_lagged_reused0,
        last_attempt_jacobian_reused0,
    )
    return jax.lax.while_loop(cond_fun, body_fun, loop_carry)


def _run_saved_loop_debug_walltime(
    *,
    step_state0,
    step_fn,
    save_n,
    t0,
    t_final,
    state_dim,
    dtype,
    max_total_steps,
    stop_after_accepted_steps=None,
    walltime_label="solver.attempt",
):
    compiled_step_fn = jax.jit(lambda step_state: step_fn(step_state, None))
    save_times = jnp.linspace(t0, t_final, save_n)
    ys_saved = jnp.zeros((save_n, state_dim), dtype=dtype)
    ts_saved = jnp.zeros((save_n,), dtype=dtype)
    dts_saved = jnp.zeros((save_n,), dtype=dtype)
    accepted_mask_saved = jnp.zeros((save_n,), dtype=bool)
    failed_mask_saved = jnp.zeros((save_n,), dtype=bool)
    fail_codes_saved = jnp.zeros((save_n,), dtype=jnp.int32)
    ys_saved = ys_saved.at[0].set(step_state0.y)
    ts_saved = ts_saved.at[0].set(t0)
    accepted_mask_saved = accepted_mask_saved.at[0].set(True)

    step_state = step_state0
    step_idx = 0
    save_idx = jnp.asarray(1, dtype=jnp.int32)

    last_attempt_accepted = jnp.asarray(False)
    last_attempt_converged = jnp.asarray(False)
    last_attempt_err_norm = jnp.asarray(jnp.inf, dtype=dtype)
    last_attempt_fail_code = jnp.asarray(0, dtype=jnp.int32)
    last_attempt_diverged = jnp.asarray(False)
    last_attempt_nonfinite_stage_state = jnp.asarray(False)
    last_attempt_nonfinite_stage_residual = jnp.asarray(False)
    last_attempt_finite_f0 = jnp.asarray(True)
    last_attempt_finite_z0 = jnp.asarray(True)
    last_attempt_finite_initial_residual = jnp.asarray(True)
    last_attempt_newton_iter_count = jnp.asarray(0, dtype=jnp.int32)
    last_attempt_final_residual_norm = jnp.asarray(jnp.inf, dtype=dtype)
    last_attempt_final_delta_norm = jnp.asarray(jnp.inf, dtype=dtype)
    last_attempt_theta_final = jnp.asarray(0.0, dtype=dtype)
    last_attempt_slow_contraction = jnp.asarray(False)
    last_attempt_residual_blowup = jnp.asarray(False)
    last_attempt_newton_nonfinite = jnp.asarray(False)
    last_attempt_lagged_reused = jnp.asarray(False)
    last_attempt_jacobian_reused = jnp.asarray(False)

    while True:
        active = bool(jax.device_get(_custom_loop_active(step_state, t_final, jnp.asarray(step_idx, dtype=jnp.int32), max_total_steps)))
        if stop_after_accepted_steps is not None:
            limit_hit = bool(jax.device_get(_accepted_step_limit_reached(step_state, stop_after_accepted_steps)))
            active = active and (not limit_hit)
        if not active:
            break

        attempt_idx = step_idx + 1
        start = time.perf_counter()
        step_state, step_info = compiled_step_fn(step_state)
        step_state = jax.block_until_ready(step_state)
        step_info = jax.block_until_ready(step_info)
        elapsed = time.perf_counter() - start
        print(f"[debug-walltime] {walltime_label} #{attempt_idx} elapsed_s={elapsed:.6f}", flush=True)

        save_idx, ys_saved, ts_saved, dts_saved, accepted_mask_saved, failed_mask_saved, fail_codes_saved = _fill_saved_slots(
            save_idx,
            save_times,
            step_info.t,
            step_info.y,
            step_info.dt,
            step_info.accepted,
            step_info.failed,
            step_info.fail_code,
            ys_saved,
            ts_saved,
            dts_saved,
            accepted_mask_saved,
            failed_mask_saved,
            fail_codes_saved,
        )

        last_attempt_accepted = jnp.asarray(step_info.accepted)
        last_attempt_converged = jnp.asarray(False if getattr(step_info, "converged", None) is None else getattr(step_info, "converged"))
        last_attempt_err_norm = jnp.asarray(jnp.inf, dtype=dtype) if getattr(step_info, "err_norm", None) is None else jnp.asarray(getattr(step_info, "err_norm"), dtype=dtype)
        last_attempt_fail_code = jnp.asarray(step_info.fail_code, dtype=jnp.int32)
        last_attempt_diverged = jnp.asarray(False if getattr(step_info, "diverged", None) is None else getattr(step_info, "diverged"))
        last_attempt_nonfinite_stage_state = jnp.asarray(False if getattr(step_info, "nonfinite_stage_state", None) is None else getattr(step_info, "nonfinite_stage_state"))
        last_attempt_nonfinite_stage_residual = jnp.asarray(False if getattr(step_info, "nonfinite_stage_residual", None) is None else getattr(step_info, "nonfinite_stage_residual"))
        last_attempt_finite_f0 = jnp.asarray(True if getattr(step_info, "finite_f0", None) is None else getattr(step_info, "finite_f0"))
        last_attempt_finite_z0 = jnp.asarray(True if getattr(step_info, "finite_z0", None) is None else getattr(step_info, "finite_z0"))
        last_attempt_finite_initial_residual = jnp.asarray(True if getattr(step_info, "finite_initial_residual", None) is None else getattr(step_info, "finite_initial_residual"))
        last_attempt_newton_iter_count = jnp.asarray(0 if getattr(step_info, "newton_iter_count", None) is None else getattr(step_info, "newton_iter_count"), dtype=jnp.int32)
        last_attempt_final_residual_norm = jnp.asarray(jnp.inf, dtype=dtype) if getattr(step_info, "final_residual_norm", None) is None else jnp.asarray(getattr(step_info, "final_residual_norm"), dtype=dtype)
        last_attempt_final_delta_norm = jnp.asarray(jnp.inf, dtype=dtype) if getattr(step_info, "final_delta_norm", None) is None else jnp.asarray(getattr(step_info, "final_delta_norm"), dtype=dtype)
        last_attempt_theta_final = jnp.asarray(0.0, dtype=dtype) if getattr(step_info, "theta_final", None) is None else jnp.asarray(getattr(step_info, "theta_final"), dtype=dtype)
        last_attempt_slow_contraction = jnp.asarray(False if getattr(step_info, "slow_contraction", None) is None else getattr(step_info, "slow_contraction"))
        last_attempt_residual_blowup = jnp.asarray(False if getattr(step_info, "residual_blowup", None) is None else getattr(step_info, "residual_blowup"))
        last_attempt_newton_nonfinite = jnp.asarray(False if getattr(step_info, "newton_nonfinite", None) is None else getattr(step_info, "newton_nonfinite"))
        last_attempt_lagged_reused = jnp.asarray(False if getattr(step_info, "lagged_reused", None) is None else getattr(step_info, "lagged_reused"))
        last_attempt_jacobian_reused = jnp.asarray(False if getattr(step_info, "jacobian_reused", None) is None else getattr(step_info, "jacobian_reused"))

        step_idx += 1

    return (
        step_state,
        jnp.asarray(step_idx, dtype=jnp.int32),
        save_idx,
        ys_saved,
        ts_saved,
        dts_saved,
        accepted_mask_saved,
        failed_mask_saved,
        fail_codes_saved,
        last_attempt_accepted,
        last_attempt_converged,
        last_attempt_err_norm,
        last_attempt_fail_code,
        last_attempt_diverged,
        last_attempt_nonfinite_stage_state,
        last_attempt_nonfinite_stage_residual,
        last_attempt_finite_f0,
        last_attempt_finite_z0,
        last_attempt_finite_initial_residual,
        last_attempt_newton_iter_count,
        last_attempt_final_residual_norm,
        last_attempt_final_delta_norm,
        last_attempt_theta_final,
        last_attempt_slow_contraction,
        last_attempt_residual_blowup,
        last_attempt_newton_nonfinite,
        last_attempt_lagged_reused,
        last_attempt_jacobian_reused,
    )


def _make_radau_stage_predictor(
    f0,
    prev_stages,
    prev_dt,
    h_value,
    c,
    dtype,
    density_size=0,
    pressure_size=0,
    er_size=0,
    prev_theta_final=None,
    prev_newton_iter_count=None,
    predictor_mode="current",
):
    base_guess = c[:, None] * f0[None, :]
    use_predictor = jnp.logical_and(
        prev_dt > 0.0,
        jnp.all(jnp.isfinite(prev_stages)),
    )
    step_ratio = h_value / jnp.maximum(prev_dt, jnp.asarray(1.0e-14, dtype=dtype))
    bounded_step_ratio = jnp.clip(step_ratio, jnp.asarray(0.25, dtype=dtype), jnp.asarray(4.0, dtype=dtype))
    prev_stage_stack = prev_stages.reshape(base_guess.shape)
    prev_stage_guess = prev_stage_stack * step_ratio
    predictor_mode_norm = str(predictor_mode).strip().lower()
    if predictor_mode_norm in {"default", "legacy"}:
        predictor_mode_norm = "current"
    predictor_aliases = {
        "gated": "dt_ratio_gated_collocation",
        "dt_gated": "dt_ratio_gated_collocation",
        "correction_gated": "collocation_correction_gated",
        "quality_gated": "newton_quality_gated_collocation",
        "dense": "ntss_dense_output",
        "dense_output": "ntss_dense_output",
        "ntss_dense": "ntss_dense_output",
        "transport": "collocation_transport_weighted",
        "transport_weighted": "collocation_transport_weighted",
        "transport_dense": "transport_weighted_dense",
        "weighted_dense": "transport_weighted_dense",
        "dense_transport": "transport_weighted_dense",
    }
    predictor_mode_norm = predictor_aliases.get(predictor_mode_norm, predictor_mode_norm)
    if predictor_mode_norm not in {
        "current",
        "collocation",
        "ntss_dense_output",
        "dt_ratio_gated_collocation",
        "collocation_correction_gated",
        "newton_quality_gated_collocation",
        "collocation_transport_weighted",
        "transport_weighted_dense",
    }:
        raise ValueError(
            "radau_predictor_mode must be one of: current, collocation, ntss_dense_output, dt_ratio_gated_collocation, collocation_correction_gated, newton_quality_gated_collocation, collocation_transport_weighted, transport_weighted_dense"
        )
    blended_guess = jnp.asarray(0.85, dtype=dtype) * prev_stage_guess + jnp.asarray(0.15, dtype=dtype) * base_guess
    prev_stage0 = prev_stage_stack[0]
    collocation_correction = c[:, None] * (f0 - prev_stage0)[None, :]
    collocation_guess = prev_stage_guess + collocation_correction
    collocation_guess = (
        jnp.asarray(0.9, dtype=dtype) * collocation_guess
        + jnp.asarray(0.1, dtype=dtype) * base_guess
    )
    dense_nodes = jnp.concatenate([c, jnp.asarray([jnp.asarray(1.0, dtype=dtype)], dtype=dtype)])
    dense_values = jnp.concatenate([prev_stage_stack, f0[None, :]], axis=0)
    dense_eval_nodes = jnp.asarray(1.0, dtype=dtype) + c * step_ratio
    dense_node_diffs = dense_nodes[:, None] - dense_nodes[None, :]
    dense_node_diffs_safe = dense_node_diffs + jnp.eye(dense_nodes.shape[0], dtype=dtype)
    dense_weights = jnp.asarray(1.0, dtype=dtype) / jnp.prod(dense_node_diffs_safe, axis=1)
    dense_eval_diffs = dense_eval_nodes[:, None] - dense_nodes[None, :]
    dense_bary = dense_weights[None, :] / dense_eval_diffs
    dense_num = dense_bary @ dense_values
    dense_den = jnp.sum(dense_bary, axis=1, keepdims=True)
    dense_guess = dense_num / dense_den
    dense_guess = jnp.where(jnp.all(jnp.isfinite(dense_guess)), dense_guess, collocation_guess)
    if density_size + pressure_size + er_size == base_guess.shape[1]:
        density_end = density_size
        pressure_end = density_size + pressure_size

        def _block_gate(block_delta, block_prev, min_gate):
            scale = jnp.maximum(jnp.maximum(jnp.abs(block_prev), jnp.abs(block_prev + block_delta)), jnp.asarray(1.0e-12, dtype=dtype))
            mismatch = jnp.sqrt(jnp.mean((block_delta / scale) ** 2) + jnp.asarray(1.0e-12, dtype=dtype))
            return min_gate + (
                (jnp.asarray(1.0, dtype=dtype) - min_gate)
                / (jnp.asarray(1.0, dtype=dtype) + (mismatch / jnp.asarray(0.5, dtype=dtype)) ** 2)
            )

        density_delta = f0[:density_end] - prev_stage0[:density_end]
        density_prev = prev_stage0[:density_end]
        pressure_delta = f0[density_end:pressure_end] - prev_stage0[density_end:pressure_end]
        pressure_prev = prev_stage0[density_end:pressure_end]
        er_delta = f0[pressure_end:pressure_end + er_size] - prev_stage0[pressure_end:pressure_end + er_size]
        er_prev = prev_stage0[pressure_end:pressure_end + er_size]

        density_gate = jnp.where(
            density_size > 0,
            _block_gate(density_delta, density_prev, jnp.asarray(0.75, dtype=dtype)),
            jnp.asarray(1.0, dtype=dtype),
        )
        pressure_gate = jnp.where(
            pressure_size > 0,
            _block_gate(pressure_delta, pressure_prev, jnp.asarray(0.85, dtype=dtype)),
            jnp.asarray(1.0, dtype=dtype),
        )
        er_gate = jnp.where(
            er_size > 0,
            _block_gate(er_delta, er_prev, jnp.asarray(0.35, dtype=dtype)),
            jnp.asarray(1.0, dtype=dtype),
        )
        block_gate_vec = jnp.concatenate(
            [
                jnp.full((density_size,), density_gate, dtype=dtype),
                jnp.full((pressure_size,), pressure_gate, dtype=dtype),
                jnp.full((er_size,), er_gate, dtype=dtype),
            ],
            axis=0,
        )
        transport_weighted_guess = prev_stage_guess + block_gate_vec[None, :] * collocation_correction
        transport_weighted_guess = (
            jnp.asarray(0.9, dtype=dtype) * transport_weighted_guess
            + jnp.asarray(0.1, dtype=dtype) * base_guess
        )
        transport_weighted_dense_guess = (
            transport_weighted_guess
            + block_gate_vec[None, :] * (dense_guess - transport_weighted_guess)
        )
        transport_weighted_dense_guess = (
            jnp.asarray(0.95, dtype=dtype) * transport_weighted_dense_guess
            + jnp.asarray(0.05, dtype=dtype) * base_guess
        )
    else:
        transport_weighted_guess = collocation_guess
        transport_weighted_dense_guess = dense_guess
    if predictor_mode_norm == "current":
        predictor_guess = blended_guess
    elif predictor_mode_norm == "collocation":
        predictor_guess = collocation_guess
    elif predictor_mode_norm == "ntss_dense_output":
        predictor_guess = dense_guess
    elif predictor_mode_norm == "transport_weighted_dense":
        predictor_guess = transport_weighted_dense_guess
    elif predictor_mode_norm == "collocation_transport_weighted":
        predictor_guess = transport_weighted_guess
    elif predictor_mode_norm == "collocation_correction_gated":
        mismatch_scale = jnp.maximum(jnp.maximum(jnp.abs(f0), jnp.abs(prev_stage0)), jnp.asarray(1.0e-12, dtype=dtype))
        mismatch_norm = jnp.sqrt(jnp.mean(((f0 - prev_stage0) / mismatch_scale) ** 2) + jnp.asarray(1.0e-12, dtype=dtype))
        correction_gate = jnp.asarray(0.25, dtype=dtype) + (
            jnp.asarray(0.75, dtype=dtype)
            / (jnp.asarray(1.0, dtype=dtype) + (mismatch_norm / jnp.asarray(0.5, dtype=dtype)) ** 2)
        )
        predictor_guess = prev_stage_guess + correction_gate * collocation_correction
        predictor_guess = (
            jnp.asarray(0.9, dtype=dtype) * predictor_guess
            + jnp.asarray(0.1, dtype=dtype) * base_guess
        )
    elif predictor_mode_norm == "newton_quality_gated_collocation":
        theta_value = jnp.asarray(0.0 if prev_theta_final is None else prev_theta_final, dtype=dtype)
        iter_value = jnp.asarray(0 if prev_newton_iter_count is None else prev_newton_iter_count, dtype=jnp.int32)
        theta_u = jnp.clip(
            (jnp.asarray(0.08, dtype=dtype) - theta_value) / jnp.asarray(0.07, dtype=dtype),
            jnp.asarray(0.0, dtype=dtype),
            jnp.asarray(1.0, dtype=dtype),
        )
        theta_gate = theta_u * theta_u * (jnp.asarray(3.0, dtype=dtype) - jnp.asarray(2.0, dtype=dtype) * theta_u)
        iter_u = jnp.clip(
            (jnp.asarray(7, dtype=jnp.int32) - iter_value).astype(dtype) / jnp.asarray(3.0, dtype=dtype),
            jnp.asarray(0.0, dtype=dtype),
            jnp.asarray(1.0, dtype=dtype),
        )
        iter_gate = iter_u * iter_u * (jnp.asarray(3.0, dtype=dtype) - jnp.asarray(2.0, dtype=dtype) * iter_u)
        quality_gate = jnp.maximum(
            jnp.asarray(0.35, dtype=dtype),
            theta_gate * iter_gate,
        )
        predictor_guess = prev_stage_guess + quality_gate * collocation_correction
        predictor_guess = (
            jnp.asarray(0.9, dtype=dtype) * predictor_guess
            + jnp.asarray(0.1, dtype=dtype) * base_guess
        )
    else:
        extrapolated_collocation_guess = (
            collocation_guess
            + (bounded_step_ratio - jnp.asarray(1.0, dtype=dtype))
            * (prev_stage_stack * bounded_step_ratio - base_guess)
        )
        extrapolated_collocation_guess = (
            jnp.asarray(0.95, dtype=dtype) * extrapolated_collocation_guess
            + jnp.asarray(0.05, dtype=dtype) * base_guess
        )
        log_step_ratio = jnp.abs(jnp.log(bounded_step_ratio))
        full_trust_ratio = jnp.asarray(jnp.log(1.1), dtype=dtype)
        zero_trust_ratio = jnp.asarray(jnp.log(1.5), dtype=dtype)
        gate_u = jnp.clip(
            (zero_trust_ratio - log_step_ratio) / (zero_trust_ratio - full_trust_ratio),
            jnp.asarray(0.0, dtype=dtype),
            jnp.asarray(1.0, dtype=dtype),
        )
        step_ratio_gate = gate_u * gate_u * (jnp.asarray(3.0, dtype=dtype) - jnp.asarray(2.0, dtype=dtype) * gate_u)
        predictor_guess = (
            collocation_guess
            + step_ratio_gate * (extrapolated_collocation_guess - collocation_guess)
        )
    return jnp.where(use_predictor, predictor_guess, base_guess).reshape((-1,))


def _apply_radau_lean_timestep_controller(
    *,
    step_state,
    trial_dt,
    trial_y,
    err_norm,
    density_err_norm,
    pressure_err_norm,
    er_err_norm,
    converged,
    stage_history,
    jacobian_out,
    cache_valid_out,
    cache_dt_out,
    cache_age_out,
    real_lu_out,
    real_piv_out,
    complex_lu_out,
    complex_piv_out,
    newton_shrink,
    diverged_final,
    nonfinite_stage_state,
    nonfinite_stage_residual,
    finite_f0,
    finite_z0,
    finite_initial_residual,
    newton_iter_count,
    final_residual_norm,
    final_delta_norm,
    theta_final,
    slow_contraction,
    residual_blowup,
    newton_nonfinite,
    lagged_reused,
    jacobian_reused,
    fail_code,
    n_accepted,
    dtype,
    dt_min,
    dt_max,
    safety_factor,
    controller_alpha,
    min_step_factor,
    max_step_factor,
    controller_mode,
    use_transport_lagged_response,
    lagged_response_reuse_mode,
    lagged_response_reuse_rtol,
    lagged_response_reuse_atol,
    project_flat,
):
    accepted = jnp.logical_and(converged, err_norm <= 1.0)
    safe_error = jnp.maximum(err_norm, 1.0e-12)
    prev_error = jnp.maximum(step_state.prev_error, 1.0e-12)
    controller_beta = jnp.asarray(0.3, dtype=dtype) * controller_alpha
    controller_gamma = jnp.asarray(0.1, dtype=dtype) * controller_alpha
    mild_growth = jnp.asarray(1.25, dtype=dtype)
    moderate_growth = jnp.asarray(1.5, dtype=dtype)
    cautious_regrowth = jnp.asarray(1.15, dtype=dtype)
    recovery_regrowth = jnp.asarray(1.35, dtype=dtype)
    strong_retry_shrink = jnp.asarray(0.6, dtype=dtype)
    moderate_retry_shrink = jnp.asarray(0.8, dtype=dtype)
    difficult_theta = jnp.asarray(0.03, dtype=dtype)
    very_difficult_theta = jnp.asarray(0.10, dtype=dtype)
    easy_theta = jnp.asarray(0.01, dtype=dtype)
    easy_error = jnp.asarray(0.05, dtype=dtype)
    recovery_theta = jnp.asarray(0.02, dtype=dtype)
    recovery_error = jnp.asarray(0.1, dtype=dtype)
    aggressive_easy_growth_floor = jnp.asarray(1.85, dtype=dtype)
    aggressive_recovery_growth_floor = jnp.asarray(1.45, dtype=dtype)
    aggressive_healthy_err = jnp.asarray(0.2, dtype=dtype)
    ntss_keep_lower = jnp.asarray(0.99, dtype=dtype)
    ntss_keep_upper = jnp.asarray(2.0, dtype=dtype)
    ntss_min_growth = jnp.asarray(0.75, dtype=dtype)
    ntss_max_growth = jnp.asarray(8.0, dtype=dtype)
    prev_dt_safe = jnp.maximum(step_state.prev_dt, dt_min)
    prev_dt_available = step_state.prev_dt > jnp.asarray(0.0, dtype=dtype)
    step_ratio_prev = jnp.where(prev_dt_available, trial_dt / prev_dt_safe, jnp.asarray(1.0, dtype=dtype))
    gustafsson_damping = jnp.where(
        prev_dt_available,
        jnp.clip(
            jnp.where(
                step_ratio_prev > jnp.asarray(1.0, dtype=dtype),
                step_ratio_prev ** (-controller_gamma),
                jnp.asarray(1.0, dtype=dtype),
            ),
            jnp.asarray(0.85, dtype=dtype),
            jnp.asarray(1.0, dtype=dtype),
        ),
        jnp.asarray(1.0, dtype=dtype),
    )
    growth_current = (
        safety_factor
        * safe_error ** (-controller_alpha)
        * prev_error ** controller_beta
        * gustafsson_damping
    )
    growth_current = jnp.clip(growth_current, min_step_factor, max_step_factor)
    growth_pi = safety_factor * safe_error ** (-controller_alpha) * prev_error ** controller_beta
    growth_pi = jnp.clip(growth_pi, min_step_factor, max_step_factor)
    growth_predictive = jnp.where(
        prev_dt_available,
        safety_factor
        * (prev_dt_safe / jnp.maximum(trial_dt, dt_min))
        * (prev_error / jnp.maximum(safe_error * safe_error, 1.0e-12)) ** controller_alpha,
        growth_pi,
    )
    growth_predictive = jnp.clip(growth_predictive, min_step_factor, max_step_factor)
    use_gustafsson_controller = controller_mode == "gustafsson"
    use_current_legacy_controller = controller_mode == "current_legacy"
    use_hairer_lean_controller = controller_mode == "hairer_lean"
    use_hairer_lean_aggressive_controller = controller_mode == "hairer_lean_aggressive"
    use_hairer_lean_transport_controller = controller_mode == "hairer_lean_transport"
    use_hairer_lean_transport_weighted_controller = controller_mode == "hairer_lean_transport_weighted"
    use_hairer_lean_transport_discounted_controller = controller_mode == "hairer_lean_transport_discounted"
    use_hairer_ntss_controller = controller_mode == "hairer_ntss"
    use_hairer_lean_family_controller = jnp.logical_or(
        jnp.logical_or(
            jnp.logical_or(
                jnp.logical_or(use_hairer_lean_controller, use_hairer_lean_aggressive_controller),
                use_hairer_lean_transport_controller,
            ),
            use_hairer_lean_transport_weighted_controller,
        ),
        use_hairer_lean_transport_discounted_controller,
    )
    growth_lean = safety_factor * safe_error ** (-controller_alpha)
    growth_lean = jnp.clip(growth_lean, min_step_factor, max_step_factor)
    nit_ref = jnp.asarray(20.0, dtype=dtype)
    newton_iter_float = jnp.asarray(newton_iter_count, dtype=dtype)
    fac_ntss = jnp.minimum(
        safety_factor,
        ((jnp.asarray(2.0, dtype=dtype) * nit_ref + jnp.asarray(1.0, dtype=dtype)) * safety_factor)
        / jnp.maximum(newton_iter_float + jnp.asarray(2.0, dtype=dtype) * nit_ref, jnp.asarray(1.0, dtype=dtype)),
    )
    quot_ntss = safe_error ** controller_alpha / jnp.maximum(fac_ntss, jnp.asarray(1.0e-12, dtype=dtype))
    quot_ntss = jnp.clip(
        quot_ntss,
        jnp.asarray(1.0, dtype=dtype) / jnp.minimum(max_step_factor, ntss_max_growth),
        jnp.asarray(1.0, dtype=dtype) / jnp.maximum(min_step_factor, ntss_min_growth),
    )
    growth_ntss = jnp.asarray(1.0, dtype=dtype) / quot_ntss
    growth_current_legacy = (
        safety_factor
        * safe_error ** (-controller_alpha)
        * prev_error ** controller_beta
    )
    growth_current_legacy = jnp.clip(growth_current_legacy, min_step_factor, max_step_factor)
    growth = jnp.where(
        use_gustafsson_controller,
        jnp.minimum(growth_pi, growth_predictive),
        jnp.where(
            use_hairer_ntss_controller,
            growth_ntss,
            jnp.where(
            use_hairer_lean_family_controller,
            growth_lean,
            jnp.where(use_current_legacy_controller, growth_current_legacy, growth_current),
            ),
        ),
    )
    difficult_accept = jnp.logical_or(
        slow_contraction,
        jnp.logical_or(
            theta_final >= difficult_theta,
            newton_iter_count >= jnp.asarray(6, dtype=jnp.int32),
        ),
    )
    very_difficult_accept = jnp.logical_or(
        theta_final >= very_difficult_theta,
        newton_iter_count >= jnp.asarray(8, dtype=jnp.int32),
    )
    easy_accept = jnp.logical_and(
        jnp.logical_and(err_norm <= easy_error, theta_final <= easy_theta),
        jnp.logical_and(newton_iter_count <= jnp.asarray(4, dtype=jnp.int32), jnp.logical_not(slow_contraction)),
    )
    recovery_ready = jnp.logical_and(
        jnp.logical_and(err_norm <= recovery_error, theta_final <= recovery_theta),
        jnp.logical_and(newton_iter_count <= jnp.asarray(5, dtype=jnp.int32), jnp.logical_not(slow_contraction)),
    )
    difficulty_growth_cap = jnp.where(
        very_difficult_accept,
        jnp.asarray(1.0, dtype=dtype),
        jnp.where(
            difficult_accept,
            mild_growth,
            moderate_growth,
        ),
    )
    post_reject_growth_cap_current = jnp.where(
        step_state.regrowth_cooldown > 0,
        jnp.where(recovery_ready, recovery_regrowth, cautious_regrowth),
        max_step_factor,
    )
    streak_growth_cap_current = jnp.where(
        step_state.easy_growth_streak >= jnp.asarray(1, dtype=jnp.int32),
        max_step_factor,
        jnp.asarray(1.75, dtype=dtype),
    )
    post_reject_growth_cap_gustafsson = jnp.where(
        step_state.recent_reject_count > 0,
        jnp.where(recovery_ready, jnp.asarray(1.45, dtype=dtype), cautious_regrowth),
        max_step_factor,
    )
    streak_growth_cap_gustafsson = max_step_factor
    post_reject_growth_cap = jnp.where(
        jnp.logical_or(use_hairer_lean_family_controller, use_hairer_ntss_controller),
        max_step_factor,
        jnp.where(
        use_gustafsson_controller,
        post_reject_growth_cap_gustafsson,
        post_reject_growth_cap_current,
        ),
    )
    streak_growth_cap = jnp.where(
        jnp.logical_or(use_hairer_lean_family_controller, use_hairer_ntss_controller),
        max_step_factor,
        jnp.where(
        use_gustafsson_controller,
        streak_growth_cap_gustafsson,
        streak_growth_cap_current,
        ),
    )
    growth_cap = jnp.where(
        jnp.logical_or(use_hairer_lean_family_controller, use_hairer_ntss_controller),
        max_step_factor,
        jnp.minimum(jnp.minimum(max_step_factor, difficulty_growth_cap), jnp.minimum(post_reject_growth_cap, streak_growth_cap)),
    )
    growth = jnp.clip(growth, min_step_factor, growth_cap)
    growth = jnp.where(
        jnp.logical_and(jnp.logical_not(jnp.logical_or(use_hairer_lean_family_controller, use_hairer_ntss_controller)), jnp.logical_and(difficult_accept, growth <= jnp.asarray(1.35, dtype=dtype))),
        jnp.asarray(1.0, dtype=dtype),
        growth,
    )
    growth = jnp.where(
        jnp.logical_and(jnp.logical_not(jnp.logical_or(use_hairer_lean_family_controller, use_hairer_ntss_controller)), jnp.logical_and(jnp.logical_and(step_state.regrowth_cooldown > 0, jnp.logical_not(use_gustafsson_controller)), recovery_ready)),
        jnp.maximum(growth, jnp.asarray(1.2, dtype=dtype)),
        growth,
    )
    aggressive_growth_ready = jnp.logical_and(
        accepted,
        jnp.logical_and(
            recovery_ready,
            jnp.logical_and(err_norm <= aggressive_healthy_err, jnp.logical_not(very_difficult_accept)),
        ),
    )
    dominant_ref = jnp.maximum(density_err_norm, pressure_err_norm)
    er_dominant = er_err_norm >= jnp.asarray(1.25, dtype=dtype) * dominant_ref
    background_easy = jnp.logical_and(
        density_err_norm <= jnp.asarray(0.15, dtype=dtype),
        pressure_err_norm <= jnp.asarray(0.15, dtype=dtype),
    )
    transport_growth_ready = jnp.logical_and(
        accepted,
        jnp.logical_and(
            er_dominant,
            jnp.logical_and(
                background_easy,
                jnp.logical_and(theta_final <= jnp.asarray(0.06, dtype=dtype), newton_iter_count <= jnp.asarray(8, dtype=jnp.int32)),
            ),
        ),
    )
    block_errs = jnp.stack(
        [
            jnp.maximum(density_err_norm, jnp.asarray(0.0, dtype=dtype)),
            jnp.maximum(pressure_err_norm, jnp.asarray(0.0, dtype=dtype)),
            jnp.maximum(er_err_norm, jnp.asarray(0.0, dtype=dtype)),
        ],
        axis=0,
    )
    block_err_total = jnp.maximum(jnp.sum(block_errs), jnp.asarray(1.0e-12, dtype=dtype))
    block_weights = block_errs / block_err_total
    block_concentration = jnp.sum(block_weights * block_weights)
    localized_u = jnp.clip(
        (block_concentration - jnp.asarray(0.5, dtype=dtype)) / jnp.asarray(0.5, dtype=dtype),
        jnp.asarray(0.0, dtype=dtype),
        jnp.asarray(1.0, dtype=dtype),
    )
    localized_gate = localized_u * localized_u * (jnp.asarray(3.0, dtype=dtype) - jnp.asarray(2.0, dtype=dtype) * localized_u)
    weighted_background_easy = jnp.logical_and(
        jnp.maximum(density_err_norm, pressure_err_norm) <= jnp.asarray(0.2, dtype=dtype),
        er_err_norm <= jnp.asarray(1.0, dtype=dtype),
    )
    weighted_transport_growth_ready = jnp.logical_and(
        accepted,
        jnp.logical_and(
            recovery_ready,
            jnp.logical_and(
                weighted_background_easy,
                jnp.logical_and(jnp.logical_not(very_difficult_accept), localized_gate >= jnp.asarray(0.35, dtype=dtype)),
            ),
        ),
    )
    weighted_localized_discount_ready = jnp.logical_and(
        localized_gate >= jnp.asarray(0.35, dtype=dtype),
        jnp.maximum(density_err_norm, pressure_err_norm) <= jnp.asarray(0.35, dtype=dtype),
    )
    growth = jnp.where(
        jnp.logical_and(use_hairer_lean_aggressive_controller, aggressive_growth_ready),
        jnp.maximum(
            growth,
            jnp.where(easy_accept, aggressive_easy_growth_floor, aggressive_recovery_growth_floor),
        ),
        growth,
    )
    weighted_growth_floor = (
        jnp.asarray(1.15, dtype=dtype)
        + jnp.asarray(0.35, dtype=dtype) * localized_gate
    )
    weighted_growth_floor = jnp.where(
        easy_accept,
        weighted_growth_floor + jnp.asarray(0.1, dtype=dtype),
        weighted_growth_floor,
    )
    growth = jnp.where(
        jnp.logical_and(use_hairer_lean_transport_weighted_controller, weighted_transport_growth_ready),
        jnp.maximum(growth, weighted_growth_floor),
        growth,
    )
    growth = jnp.where(
        jnp.logical_and(use_hairer_lean_transport_controller, transport_growth_ready),
        jnp.maximum(
            growth,
            jnp.where(
                easy_accept,
                jnp.asarray(1.75, dtype=dtype),
                jnp.asarray(1.35, dtype=dtype),
            ),
        ),
        growth,
    )
    growth = jnp.where(
        jnp.logical_and(
            accepted,
            jnp.logical_and(
                use_hairer_ntss_controller,
                jnp.logical_and(growth >= ntss_keep_lower, growth <= ntss_keep_upper),
            ),
        ),
        jnp.asarray(1.0, dtype=dtype),
        growth,
    )
    next_dt = jnp.clip(trial_dt * growth, dt_min, dt_max)

    def _accept(_):
        t_new = step_state.t + trial_dt
        accepted_y = _project_flat_state_if_needed(trial_y, project_flat)
        next_dt_accept = jnp.clip(trial_dt * growth, dt_min, dt_max)
        lagged_reuse_global = lagged_response_reuse_mode == "global_state_drift"
        lagged_reuse_metric = _lagged_response_global_reuse_metric(
            accepted_y,
            step_state.lagged_reference_y,
            atol=lagged_response_reuse_atol,
            rtol=lagged_response_reuse_rtol,
        )
        keep_lagged_response = jnp.logical_and(
            jnp.asarray(use_transport_lagged_response),
            jnp.logical_and(
                lagged_reuse_global,
                lagged_reuse_metric <= jnp.asarray(1.0, dtype=dtype),
            ),
        )
        status_next = jnp.asarray([0, fail_code, n_accepted + 1], dtype=jnp.int32)
        regrowth_cooldown_next = jnp.where(
            jnp.logical_or(use_hairer_lean_family_controller, use_hairer_ntss_controller),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.where(
            use_gustafsson_controller,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.maximum(
                step_state.regrowth_cooldown - jnp.where(easy_accept, jnp.asarray(2, dtype=jnp.int32), jnp.asarray(1, dtype=jnp.int32)),
                jnp.asarray(0, dtype=jnp.int32),
            ),
            ),
        )
        easy_growth_streak_next = jnp.where(
            jnp.logical_or(use_hairer_lean_family_controller, use_hairer_ntss_controller),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.where(
            use_gustafsson_controller,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.where(
                easy_accept,
                jnp.minimum(step_state.easy_growth_streak + jnp.asarray(1, dtype=jnp.int32), jnp.asarray(3, dtype=jnp.int32)),
                jnp.asarray(0, dtype=jnp.int32),
            ),
            ),
        )
        return _RadauStepState(
            t=t_new,
            y=accepted_y,
            dt=next_dt_accept,
            status=status_next,
            prev_error=safe_error,
            prev_stages=stage_history,
            prev_dt=trial_dt,
            recent_reject_count=jnp.asarray(0, dtype=jnp.int32),
            regrowth_cooldown=regrowth_cooldown_next,
            easy_growth_streak=easy_growth_streak_next,
            lagged_response_cache=step_state.lagged_response_cache,
            lagged_response_valid=keep_lagged_response,
            lagged_reference_y=step_state.lagged_reference_y,
            jacobian=jacobian_out,
            cache_valid=cache_valid_out,
            cache_dt=cache_dt_out,
            cache_age=cache_age_out,
            real_lu=real_lu_out,
            real_piv=real_piv_out,
            complex_lu=complex_lu_out,
            complex_piv=complex_piv_out,
            prev_theta_final=theta_final,
            prev_newton_iter_count=newton_iter_count,
        ), _RadauStepInfo(
            y=accepted_y,
            t=t_new,
            dt=trial_dt,
            next_dt=next_dt_accept,
            growth=growth,
            lagged_reused=lagged_reused,
            jacobian_reused=jacobian_reused,
            accepted=jnp.asarray(True),
            failed=jnp.asarray(False),
            fail_code=fail_code,
            converged=converged,
            err_norm=err_norm,
            diverged=diverged_final,
            nonfinite_stage_state=nonfinite_stage_state,
            nonfinite_stage_residual=nonfinite_stage_residual,
            finite_f0=finite_f0,
            finite_z0=finite_z0,
            finite_initial_residual=finite_initial_residual,
            newton_iter_count=newton_iter_count,
            final_residual_norm=final_residual_norm,
            final_delta_norm=final_delta_norm,
            theta_final=theta_final,
            slow_contraction=slow_contraction,
            residual_blowup=residual_blowup,
            newton_nonfinite=newton_nonfinite,
        )

    def _reject(_):
        code = jnp.where(
            converged,
            jnp.asarray(2, dtype=jnp.int32),
            jnp.asarray(1, dtype=jnp.int32),
        )
        retry_count_next = jnp.minimum(step_state.recent_reject_count + jnp.asarray(1, dtype=jnp.int32), jnp.asarray(3, dtype=jnp.int32))
        retry_shrink = jnp.where(
            retry_count_next >= jnp.asarray(3, dtype=jnp.int32),
            strong_retry_shrink,
            jnp.where(
                retry_count_next >= jnp.asarray(2, dtype=jnp.int32),
                moderate_retry_shrink,
                jnp.asarray(1.0, dtype=dtype),
            ),
        )
        localized_retry_shrink = jnp.maximum(
            retry_shrink,
            jnp.asarray(0.85, dtype=dtype) + jnp.asarray(0.1, dtype=dtype) * localized_gate,
        )
        retry_shrink = jnp.where(
            jnp.logical_and(use_hairer_lean_transport_discounted_controller, weighted_localized_discount_ready),
            localized_retry_shrink,
            retry_shrink,
        )
        reject_cap = jnp.where(
            jnp.logical_and(use_hairer_lean_transport_discounted_controller, weighted_localized_discount_ready),
            trial_dt * (jnp.asarray(0.6, dtype=dtype) + jnp.asarray(0.2, dtype=dtype) * localized_gate),
            trial_dt * jnp.asarray(0.5, dtype=dtype),
        )
        reduced_dt_base = jnp.maximum(
            jnp.minimum(jnp.where(converged, next_dt, trial_dt * newton_shrink), reject_cap),
            dt_min,
        )
        reduced_dt = jnp.maximum(reduced_dt_base * retry_shrink, dt_min)
        status_next = jnp.asarray([0, 0, n_accepted], dtype=jnp.int32)
        fail_now = jnp.logical_and(reduced_dt <= dt_min * (1.0 + 1.0e-12), jnp.logical_not(accepted))
        fail_code_next = jnp.where(fail_now, code, jnp.asarray(0, dtype=jnp.int32))
        status_next = status_next.at[0].set(fail_now.astype(jnp.int32))
        status_next = status_next.at[1].set(fail_code_next)
        return _RadauStepState(
            t=step_state.t,
            y=step_state.y,
            dt=reduced_dt,
            status=status_next,
            prev_error=step_state.prev_error,
            prev_stages=step_state.prev_stages,
            prev_dt=step_state.prev_dt,
            recent_reject_count=retry_count_next,
            regrowth_cooldown=jnp.where(
                jnp.logical_or(use_gustafsson_controller, use_hairer_ntss_controller),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(2, dtype=jnp.int32),
            ),
            easy_growth_streak=jnp.asarray(0, dtype=jnp.int32),
            lagged_response_cache=step_state.lagged_response_cache,
            lagged_response_valid=step_state.lagged_response_valid,
            lagged_reference_y=step_state.lagged_reference_y,
            jacobian=jacobian_out,
            cache_valid=cache_valid_out,
            cache_dt=cache_dt_out,
            cache_age=cache_age_out,
            real_lu=real_lu_out,
            real_piv=real_piv_out,
            complex_lu=complex_lu_out,
            complex_piv=complex_piv_out,
            prev_theta_final=theta_final,
            prev_newton_iter_count=newton_iter_count,
        ), _RadauStepInfo(
            y=step_state.y,
            t=step_state.t,
            dt=trial_dt,
            next_dt=reduced_dt,
            growth=jnp.where(trial_dt > 0, reduced_dt / trial_dt, jnp.asarray(1.0, dtype=dtype)),
            lagged_reused=lagged_reused,
            jacobian_reused=jacobian_reused,
            accepted=jnp.asarray(False),
            failed=fail_now,
            fail_code=fail_code_next,
            converged=converged,
            err_norm=err_norm,
            diverged=diverged_final,
            nonfinite_stage_state=nonfinite_stage_state,
            nonfinite_stage_residual=nonfinite_stage_residual,
            finite_f0=finite_f0,
            finite_z0=finite_z0,
            finite_initial_residual=finite_initial_residual,
            newton_iter_count=newton_iter_count,
            final_residual_norm=final_residual_norm,
            final_delta_norm=final_delta_norm,
            theta_final=theta_final,
            slow_contraction=slow_contraction,
            residual_blowup=residual_blowup,
            newton_nonfinite=newton_nonfinite,
        )

    return jax.lax.cond(accepted, _accept, _reject, operand=None)

@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class _RadauSolverConfig(TransportSolver):
    t0: float
    t1: float
    dt: float
    rtol: float = 1.0e-6
    atol: float = 1.0e-8
    max_step: float = 1.0
    min_step: float = 1.0e-14
    tol: float = 1.0e-8
    maxiter: int = 20
    error_estimator: str = "embedded2"
    num_stages: int = 3
    safety_factor: float = 0.9
    min_step_factor: float = 0.1
    max_step_factor: float = 5.0
    jacobian_reuse_rtol: float = 0.1
    max_jacobian_age: int = 8
    rhs_mode: str = "black_box"
    newton_divergence_mode: str = "legacy"
    newton_residual_norm: str = "raw"
    newton_tol_mode: str = "residual"
    newton_fnewt_mode: str = "tol"
    controller_mode: str = "current"
    predictor_mode: str = "current"
    lagged_response_reuse_mode: str = "retry_only"
    lagged_response_reuse_rtol: float = 5.0e-2
    lagged_response_reuse_atol: float = 1.0e-8
    max_steps: int = 20000
    stop_after_accepted_steps: int | None = None
    n_steps: int = 0
    debug_stage_markers: bool = False
    debug_walltime_attempts: bool = False

    def __init__(
        self,
        t0: float = 0.0,
        t1: float = 1.0,
        dt: float = 1.0e-2,
        rtol: float = 1.0e-6,
        atol: float = 1.0e-8,
        max_step: float = 1.0,
        min_step: float = 1.0e-14,
        tol: float = 1.0e-8,
        maxiter: int = 20,
        error_estimator: str = "embedded2",
        num_stages: int = 3,
        safety_factor: float = 0.9,
        min_step_factor: float = 0.1,
        max_step_factor: float = 5.0,
        jacobian_reuse_rtol: float = 0.1,
        max_jacobian_age: int = 8,
        rhs_mode: str = "black_box",
        newton_divergence_mode: str = "legacy",
        newton_residual_norm: str = "raw",
        newton_tol_mode: str = "residual",
        newton_fnewt_mode: str = "tol",
        controller_mode: str = "current",
        predictor_mode: str = "current",
        lagged_response_reuse_mode: str = "retry_only",
        lagged_response_reuse_rtol: float = 5.0e-2,
        lagged_response_reuse_atol: float = 1.0e-8,
        max_steps: int = 20000,
        stop_after_accepted_steps: int | None = None,
        debug_stage_markers: bool = False,
        debug_walltime_attempts: bool = False,
        save_n=None,
    ):
        n_steps = max(1, int(jnp.ceil((float(t1) - float(t0)) / float(dt))))
        object.__setattr__(self, "t0", float(t0))
        object.__setattr__(self, "t1", float(t1))
        object.__setattr__(self, "dt", float(dt))
        object.__setattr__(self, "rtol", float(rtol))
        object.__setattr__(self, "atol", float(atol))
        object.__setattr__(self, "max_step", float(max_step))
        object.__setattr__(self, "min_step", float(min_step))
        object.__setattr__(self, "tol", float(tol))
        object.__setattr__(self, "maxiter", int(max(1, maxiter)))
        error_estimator_norm = str(error_estimator).strip().lower()
        error_estimator_aliases = {
            "ntss": "embedded2_ntss_scale",
            "ntss_scale": "embedded2_ntss_scale",
            "hairer": "embedded2_ntss_scale",
            "ntss_max": "embedded2_ntss_max_scale",
            "ntss_blend": "embedded2_ntss_blend_scale",
            "transport": "embedded2_ntss_transport_scale",
            "ntss_transport": "embedded2_ntss_transport_scale",
            "block_floor": "embedded2_ntss_block_floor_scale",
            "floor_scale": "embedded2_ntss_block_floor_scale",
            "ntss_block_floor": "embedded2_ntss_block_floor_scale",
            "block_rms": "embedded2_ntss_block_rms",
            "ntss_block_rms": "embedded2_ntss_block_rms",
        }
        error_estimator_norm = error_estimator_aliases.get(error_estimator_norm, error_estimator_norm)
        if error_estimator_norm not in {
            "embedded2",
            "embedded2_mean_scale",
            "embedded2_blend_scale",
            "embedded2_ntss_scale",
            "embedded2_ntss_max_scale",
            "embedded2_ntss_blend_scale",
            "embedded2_ntss_transport_scale",
            "embedded2_ntss_block_floor_scale",
            "embedded2_ntss_block_rms",
        }:
            raise ValueError(
                "radau_error_estimator must be one of: embedded2, embedded2_mean_scale, embedded2_blend_scale, embedded2_ntss_scale, embedded2_ntss_max_scale, embedded2_ntss_blend_scale, embedded2_ntss_transport_scale, embedded2_ntss_block_floor_scale, embedded2_ntss_block_rms"
            )
        object.__setattr__(self, "error_estimator", error_estimator_norm)
        object.__setattr__(self, "num_stages", int(num_stages))
        object.__setattr__(self, "safety_factor", float(safety_factor))
        object.__setattr__(self, "min_step_factor", float(min_step_factor))
        object.__setattr__(self, "max_step_factor", float(max_step_factor))
        object.__setattr__(self, "jacobian_reuse_rtol", float(jacobian_reuse_rtol))
        object.__setattr__(self, "max_jacobian_age", int(max(0, max_jacobian_age)))
        object.__setattr__(self, "rhs_mode", str(rhs_mode).strip().lower())
        object.__setattr__(self, "newton_divergence_mode", str(newton_divergence_mode).strip().lower())
        object.__setattr__(self, "newton_residual_norm", str(newton_residual_norm).strip().lower())
        object.__setattr__(self, "newton_tol_mode", str(newton_tol_mode).strip().lower())
        object.__setattr__(self, "newton_fnewt_mode", str(newton_fnewt_mode).strip().lower())
        controller_mode_norm = str(controller_mode).strip().lower()
        controller_aliases = {
            "default": "current",
            "legacy": "current",
            "heuristic": "current",
            "lean": "current",
            "old_current": "current_legacy",
            "pre_tuned": "current_legacy",
            "ntss_quality": "current_legacy",
            "original": "hairer_lean",
            "simple": "hairer_lean",
            "aggressive": "hairer_lean_aggressive",
            "lean_aggressive": "hairer_lean_aggressive",
            "healthy_regrowth": "hairer_lean_aggressive",
            "transport": "hairer_lean_transport",
            "lean_transport": "hairer_lean_transport",
            "transport_aware": "hairer_lean_transport",
            "transport_weighted": "hairer_lean_transport_weighted",
            "lean_transport_weighted": "hairer_lean_transport_weighted",
            "transport_symmetric": "hairer_lean_transport_weighted",
            "discounted": "hairer_lean_transport_discounted",
            "transport_discounted": "hairer_lean_transport_discounted",
            "lean_transport_discounted": "hairer_lean_transport_discounted",
            "ntss": "hairer_ntss",
            "hairer": "hairer_ntss",
            "hairer_default": "hairer_ntss",
            "ntss_default": "hairer_ntss",
            "pi": "gustafsson",
            "predictive": "gustafsson",
            "standard": "gustafsson",
        }
        controller_mode_norm = controller_aliases.get(controller_mode_norm, controller_mode_norm)
        if controller_mode_norm not in {"current", "current_legacy", "gustafsson", "hairer_lean", "hairer_lean_aggressive", "hairer_lean_transport", "hairer_lean_transport_weighted", "hairer_lean_transport_discounted", "hairer_ntss"}:
            raise ValueError(
                "radau_controller_mode must be one of: current, current_legacy, gustafsson, hairer_lean, hairer_lean_aggressive, hairer_lean_transport, hairer_lean_transport_weighted, hairer_lean_transport_discounted, hairer_ntss"
            )
        object.__setattr__(self, "controller_mode", controller_mode_norm)
        predictor_mode_norm = str(predictor_mode).strip().lower()
        predictor_aliases = {
            "default": "current",
            "legacy": "current",
            "stage_history": "current",
            "hairer": "collocation",
            "ntss": "collocation",
            "dense": "ntss_dense_output",
            "dense_output": "ntss_dense_output",
            "ntss_dense": "ntss_dense_output",
            "transport": "collocation_transport_weighted",
            "transport_weighted": "collocation_transport_weighted",
            "transport_dense": "transport_weighted_dense",
            "weighted_dense": "transport_weighted_dense",
            "dense_transport": "transport_weighted_dense",
            "gated": "dt_ratio_gated_collocation",
            "dt_gated": "dt_ratio_gated_collocation",
            "correction_gated": "collocation_correction_gated",
            "quality_gated": "newton_quality_gated_collocation",
        }
        predictor_mode_norm = predictor_aliases.get(predictor_mode_norm, predictor_mode_norm)
        if predictor_mode_norm not in {
            "current",
            "collocation",
            "ntss_dense_output",
            "collocation_transport_weighted",
            "transport_weighted_dense",
            "dt_ratio_gated_collocation",
            "collocation_correction_gated",
            "newton_quality_gated_collocation",
        }:
            raise ValueError(
                "radau_predictor_mode must be one of: current, collocation, ntss_dense_output, collocation_transport_weighted, transport_weighted_dense, dt_ratio_gated_collocation, collocation_correction_gated, newton_quality_gated_collocation"
            )
        object.__setattr__(self, "predictor_mode", predictor_mode_norm)
        lagged_reuse_mode_norm = str(lagged_response_reuse_mode).strip().lower()
        lagged_reuse_aliases = {
            "default": "retry_only",
            "retry": "retry_only",
            "state": "global_state_drift",
            "global": "global_state_drift",
            "state_drift": "global_state_drift",
        }
        lagged_reuse_mode_norm = lagged_reuse_aliases.get(lagged_reuse_mode_norm, lagged_reuse_mode_norm)
        if lagged_reuse_mode_norm not in {"retry_only", "global_state_drift"}:
            raise ValueError(
                "lagged_response_reuse_mode must be one of: retry_only, global_state_drift"
            )
        object.__setattr__(self, "lagged_response_reuse_mode", lagged_reuse_mode_norm)
        object.__setattr__(self, "lagged_response_reuse_rtol", float(lagged_response_reuse_rtol))
        object.__setattr__(self, "lagged_response_reuse_atol", float(lagged_response_reuse_atol))
        object.__setattr__(self, "max_steps", int(max(1, max_steps)))
        if stop_after_accepted_steps is not None:
            stop_after_accepted_steps = int(max(1, stop_after_accepted_steps))
        object.__setattr__(self, "stop_after_accepted_steps", stop_after_accepted_steps)
        object.__setattr__(self, "n_steps", n_steps)
        object.__setattr__(self, "debug_stage_markers", bool(debug_stage_markers))
        object.__setattr__(self, "debug_walltime_attempts", bool(debug_walltime_attempts))
        object.__setattr__(self, "save_n", save_n)

@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class _RadauAcceptedStepCarry:
    t: Any
    y: Any
    dt: Any
    prev_error: Any
    prev_stages: Any
    prev_dt: Any
    recent_reject_count: Any
    regrowth_cooldown: Any
    easy_growth_streak: Any
    lagged_response_cache: Any
    lagged_response_valid: Any
    lagged_reference_y: Any
    jacobian: Any
    cache_valid: Any
    cache_dt: Any
    cache_age: Any
    real_lu: Any
    real_piv: Any
    complex_lu: Any
    complex_piv: Any
    prev_theta_final: Any
    prev_newton_iter_count: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class _RadauAcceptedStepAttemptResult:
    carry_after_attempt: Any
    trial_dt: Any
    trial_y: Any
    err_norm: Any
    converged: Any
    stage_history: Any
    jacobian_out: Any
    cache_valid_out: Any
    cache_dt_out: Any
    cache_age_out: Any
    real_lu_out: Any
    real_piv_out: Any
    complex_lu_out: Any
    complex_piv_out: Any
    newton_shrink: Any
    diverged_final: Any
    nonfinite_stage_state: Any
    nonfinite_stage_residual: Any
    finite_f0: Any
    finite_z0: Any
    finite_initial_residual: Any
    newton_iter_count: Any
    final_residual_norm: Any
    final_delta_norm: Any
    theta_final: Any
    slow_contraction_final: Any
    residual_blowup_final: Any
    newton_nonfinite_final: Any
    density_err_norm: Any
    pressure_err_norm: Any
    er_err_norm: Any
    lagged_response_reused: Any
    jacobian_reused: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class _RadauAcceptedStepBackwardPayloadCandidate:
    t_start: Any
    y_start: Any
    dt: Any
    prev_stages: Any
    prev_dt: Any
    prev_theta_final: Any
    prev_newton_iter_count: Any
    lagged_response_cache: Any
    lagged_response_valid: Any
    lagged_reference_y: Any
    y_end: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class _RadauAcceptedStepAttemptContext:
    t_final: Any
    use_transport_lagged_response: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class _RadauAcceptedRolloutResult:
    final_carry: Any
    final_y: Any
    trial_ys: Any
    err_norms: Any
    converged_mask: Any
    accepted_dts: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class _RadauControllerRolloutResult:
    final_step_state: Any
    final_carry: Any
    step_ys: Any
    err_norms: Any
    accepted_mask: Any
    attempted_dts: Any
    next_dts: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class _RadauAdaptiveRolloutTrace:
    accepted_mask: Any
    active_mask: Any
    t_start: Any
    y_start: Any
    dt: Any
    prev_stages: Any
    prev_dt: Any
    prev_theta_final: Any
    prev_newton_iter_count: Any
    lagged_response_cache: Any
    lagged_response_valid: Any
    lagged_reference_y: Any
    y_end: Any
    err_norms: Any
    theta_finals: Any
    newton_iter_counts: Any
    cache_valid_next: Any
    attempted_dts: Any
    next_dts: Any
    step_ts: Any
    next_recent_reject_count: Any
    next_regrowth_cooldown: Any
    next_easy_growth_streak: Any
    next_lagged_response_valid: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class _RadauAdaptiveRolloutResult:
    final_step_state: Any
    final_carry: Any
    trace: _RadauAdaptiveRolloutTrace
    attempt_count: Any
    accepted_count: Any
    completed: Any
    failed: Any
    fail_code: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class _RadauReplayNanDiagnostic:
    tangent_finite_mask: Any
    first_bad_index: Any
    first_bad_was_accepted: Any
    first_bad_dt: Any
    final_tangent_finite: Any
    dt_dot_abs: Any
    prev_error_dot_abs: Any
    density_dot_max_abs: Any
    pressure_dot_max_abs: Any
    er_dot_max_abs: Any
    y_dot_max_abs: Any
    prev_stages_dot_max_abs: Any
    lagged_response_cache_dot_max_abs: Any
    jacobian_dot_max_abs: Any
    real_lu_dot_max_abs: Any
    complex_lu_dot_max_abs: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class _RadauAcceptedStepZeroTangentComparison:
    target_attempt_index: Any
    target_was_accepted: Any
    trial_dt: Any
    custom_trial_y_max_abs: Any
    custom_stage_history_max_abs: Any
    custom_finite: Any
    direct_trial_y_max_abs: Any
    direct_stage_history_max_abs: Any
    direct_finite: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class _RadauAcceptedStepMapResult:
    next_carry: Any
    accepted_y: Any
    accepted_t: Any
    accepted_dt: Any
    err_norm: Any
    converged: Any
    stage_history: Any
    theta_final: Any
    newton_iter_count: Any
    final_residual_norm: Any
    final_delta_norm: Any
    density_err_norm: Any
    pressure_err_norm: Any
    er_err_norm: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class _PreparedRadauAcceptedRollout:
    kernel_context: Any
    physics_context: Any
    initial_carry: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class _RadauSolveExecutionContext:
    kernel_context: Any
    physics_context: Any
    attempt_context: Any
    dtype: Any
    dt_min: Any
    dt_max: Any
    safety_factor: Any
    controller_alpha: Any
    min_step_factor: Any
    max_step_factor: Any
    controller_mode: Any
    use_transport_lagged_response: Any
    lagged_response_reuse_mode: Any
    lagged_response_reuse_rtol: Any
    lagged_response_reuse_atol: Any
    project_flat: Any
    debug_newton_trace: Any


def _build_prepared_radau_execution_context(
    *,
    solver: "RADAUSolver",
    prepared_rollout: _PreparedRadauAcceptedRollout,
) -> _RadauSolveExecutionContext:
    dtype = prepared_rollout.kernel_context.dtype
    if solver.num_stages not in _RADAU_STAGE_CONFIGS:
        raise ValueError(
            f"Unsupported custom Radau stage count '{solver.num_stages}'. "
            f"Available options: {sorted(_RADAU_STAGE_CONFIGS)}"
        )
    stage_cfg = _RADAU_STAGE_CONFIGS[solver.num_stages]
    error_order = float(stage_cfg.embedded_order if stage_cfg.has_embedded_estimator else stage_cfg.order)
    controller_alpha = 0.7 / (error_order + 1.0)
    return _RadauSolveExecutionContext(
        kernel_context=prepared_rollout.kernel_context,
        physics_context=prepared_rollout.physics_context,
        attempt_context=_RadauAcceptedStepAttemptContext(
            t_final=jnp.asarray(solver.t1, dtype=dtype),
            use_transport_lagged_response=jnp.asarray(prepared_rollout.kernel_context.use_transport_lagged_response),
        ),
        dtype=dtype,
        dt_min=jnp.asarray(solver.min_step, dtype=dtype),
        dt_max=jnp.asarray(solver.max_step, dtype=dtype),
        safety_factor=solver.safety_factor,
        controller_alpha=controller_alpha,
        min_step_factor=solver.min_step_factor,
        max_step_factor=solver.max_step_factor,
        controller_mode=str(getattr(solver, "controller_mode", "current")).strip().lower(),
        use_transport_lagged_response=prepared_rollout.kernel_context.use_transport_lagged_response,
        lagged_response_reuse_mode=str(getattr(solver, "lagged_response_reuse_mode", "retry_only")).strip().lower(),
        lagged_response_reuse_rtol=jnp.asarray(getattr(solver, "lagged_response_reuse_rtol", 5.0e-2), dtype=dtype),
        lagged_response_reuse_atol=jnp.asarray(getattr(solver, "lagged_response_reuse_atol", 1.0e-8), dtype=dtype),
        project_flat=prepared_rollout.physics_context.project_flat,
        debug_newton_trace=bool(getattr(solver, "debug_stage_markers", False)),
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class _RadauAcceptedStepTangentInputs:
    dy: Any
    dh: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class _RadauAcceptedStepApproximateTangentResult:
    dy_next: Any
    dz_stages: Any
    dtrial_dt: Any
    dtrial_y: Any
    dstage_history: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class _RadauStageSubsolveResult:
    iter_final: Any
    z_final: Any
    delta_norm_final: Any
    newton_metric_final: Any
    theta_final: Any
    diverged_final: Any
    shrink_suggest_final: Any
    slow_contraction_final: Any
    residual_blowup_final: Any
    newton_nonfinite_final: Any
    finite_initial_residual: Any
    nonfinite_stage_state: Any
    nonfinite_stage_residual: Any
    final_residual: Any
    final_residual_norm: Any
    converged: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class _RadauStageSubsolveInputs:
    flat_y: Any
    t_value: Any
    h_value: Any
    z0: Any
    f0: Any
    jacobian_ref: Any
    lagged_response: Any
    real_lu_out: Any
    real_piv_out: Any
    complex_lu_out: Any
    complex_piv_out: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class _RadauStageSubsolveTangentInputs:
    dy: Any
    dh: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class _RadauStageSubsolveApproximateTangentResult:
    dz_stages: Any
    dz_flat: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class _RadauAcceptedStepKernelContext:
    radau_transform: Any
    radau_inv_transform: Any
    c: Any
    a: Any
    b: Any
    b_error: Any
    embedded_f0_weight: Any
    dtype: Any
    density_size: Any
    pressure_size: Any
    er_size: Any
    predictor_mode: Any
    jacobian_reuse_rtol: Any
    use_lagged_linear_response: Any
    num_stages: Any
    state_dim: Any
    num_complex_pairs: Any
    complex_dim: Any
    identity_2: Any
    identity_n: Any
    use_rms_residual_norm: Any
    residual_size_sqrt: Any
    stage_scale: Any
    radau_real_eig: Any
    radau_complex_blocks: Any
    predictor_fnewt: Any
    maxiter: Any
    tol: Any
    error_scale_mode: Any
    atol: Any
    rtol: Any
    estimator_rtol_eff: Any
    use_hairer_newton_tol: Any
    use_hairer_scaled_correction: Any
    theta_diverge_threshold: Any
    predictor_defect_floor: Any
    predictor_defect_cap: Any
    residual_blowup_factor: Any
    theta_clip_min: Any
    theta_clip_max: Any
    newton_shrink_num: Any
    newton_shrink_min: Any
    newton_shrink_max: Any
    tiny_scalar: Any
    zero_scalar: Any
    debug_newton_trace: Any
    use_transport_lagged_response: Any


@dataclasses.dataclass(frozen=True, eq=False)
class _RadauAcceptedStepPhysicsContext:
    unpack_flat: Callable[[Any], Any]
    project_flat: Callable[[Any], Any] | None
    build_lagged_response: Callable[[Any], Any] | None
    flat_rhs: Callable[[Any, Any], Any]
    flat_rhs_with_lagged_response: Callable[[Any, Any, Any], Any]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class _RadauStepState:
    t: Any
    y: Any
    dt: Any
    status: Any
    prev_error: Any
    prev_stages: Any
    prev_dt: Any
    recent_reject_count: Any
    regrowth_cooldown: Any
    easy_growth_streak: Any
    lagged_response_cache: Any
    lagged_response_valid: Any
    lagged_reference_y: Any
    jacobian: Any
    cache_valid: Any
    cache_dt: Any
    cache_age: Any
    real_lu: Any
    real_piv: Any
    complex_lu: Any
    complex_piv: Any
    prev_theta_final: Any
    prev_newton_iter_count: Any


def _radau_carry_from_step_state(step_state: "_RadauStepState") -> _RadauAcceptedStepCarry:
    """Extract carry needed to describe the realized accepted Radau step."""
    return _RadauAcceptedStepCarry(
        t=step_state.t,
        y=step_state.y,
        dt=step_state.dt,
        prev_error=step_state.prev_error,
        prev_stages=step_state.prev_stages,
        prev_dt=step_state.prev_dt,
        recent_reject_count=step_state.recent_reject_count,
        regrowth_cooldown=step_state.regrowth_cooldown,
        easy_growth_streak=step_state.easy_growth_streak,
        lagged_response_cache=step_state.lagged_response_cache,
        lagged_response_valid=step_state.lagged_response_valid,
        lagged_reference_y=step_state.lagged_reference_y,
        jacobian=step_state.jacobian,
        cache_valid=step_state.cache_valid,
        cache_dt=step_state.cache_dt,
        cache_age=step_state.cache_age,
        real_lu=step_state.real_lu,
        real_piv=step_state.real_piv,
        complex_lu=step_state.complex_lu,
        complex_piv=step_state.complex_piv,
        prev_theta_final=step_state.prev_theta_final,
        prev_newton_iter_count=step_state.prev_newton_iter_count,
    )


def _radau_step_state_from_carry(carry: _RadauAcceptedStepCarry, *, status) -> _RadauStepState:
    """Rebuild full step state from carry plus controller status bookkeeping."""
    return _RadauStepState(
        t=carry.t,
        y=carry.y,
        dt=carry.dt,
        status=status,
        prev_error=carry.prev_error,
        prev_stages=carry.prev_stages,
        prev_dt=carry.prev_dt,
        recent_reject_count=carry.recent_reject_count,
        regrowth_cooldown=carry.regrowth_cooldown,
        easy_growth_streak=carry.easy_growth_streak,
        lagged_response_cache=carry.lagged_response_cache,
        lagged_response_valid=carry.lagged_response_valid,
        lagged_reference_y=carry.lagged_reference_y,
        jacobian=carry.jacobian,
        cache_valid=carry.cache_valid,
        cache_dt=carry.cache_dt,
        cache_age=carry.cache_age,
        real_lu=carry.real_lu,
        real_piv=carry.real_piv,
        complex_lu=carry.complex_lu,
        complex_piv=carry.complex_piv,
        prev_theta_final=carry.prev_theta_final,
        prev_newton_iter_count=carry.prev_newton_iter_count,
    )


def _radau_backward_payload_candidate(
    carry_in: _RadauAcceptedStepCarry,
    attempt_result: _RadauAcceptedStepAttemptResult,
) -> _RadauAcceptedStepBackwardPayloadCandidate:
    """Candidate minimal payload for future accepted-step backward rules.

    This is intentionally conservative: it keeps predictor and lagged-response
    context, but excludes controller bookkeeping and recomputable Jacobian/LU
    caches.
    """
    return _RadauAcceptedStepBackwardPayloadCandidate(
        t_start=carry_in.t,
        y_start=carry_in.y,
        dt=attempt_result.trial_dt,
        prev_stages=carry_in.prev_stages,
        prev_dt=carry_in.prev_dt,
        prev_theta_final=carry_in.prev_theta_final,
        prev_newton_iter_count=carry_in.prev_newton_iter_count,
        lagged_response_cache=carry_in.lagged_response_cache,
        lagged_response_valid=carry_in.lagged_response_valid,
        lagged_reference_y=carry_in.lagged_reference_y,
        y_end=attempt_result.trial_y,
    )


def _radau_extract_tangent_inputs_from_carry(
    carry_tangent: _RadauAcceptedStepCarry,
) -> _RadauAcceptedStepTangentInputs:
    """Extract the first tangent-active accepted-step inputs from a carry tangent.

    The first implicit-diff JVP will treat the accepted-step map as primarily
    active with respect to:

    - `y_n`
    - `h`

    while most controller/caching state remains forward-only.
    """
    return _RadauAcceptedStepTangentInputs(
        dy=carry_tangent.y,
        dh=carry_tangent.dt,
    )


def _radau_build_approximate_tangent_result(
    tangent_inputs: _RadauAcceptedStepTangentInputs,
    tangent_result: _RadauAcceptedStepApproximateTangentResult,
    *,
    attempt_result: _RadauAcceptedStepAttemptResult,
) -> _RadauAcceptedStepAttemptResult:
    """Lift the approximate accepted-step tangent back to attempt-result shape.

    This is a forward-neutral structural helper for the future JVP rule. Only
    the fields controlled by the accepted-step implicit map are populated with
    nonzero tangents here; diagnostic/controller outputs remain zero-like.
    """
    def _zero_tangent_like(x):
        arr = jnp.asarray(x)
        if jnp.issubdtype(arr.dtype, jnp.inexact):
            return jnp.zeros_like(arr)
        return jnp.zeros(arr.shape, dtype=jax.dtypes.float0)

    carry_after_attempt_tangent = dataclasses.replace(
        jax.tree_util.tree_map(_zero_tangent_like, attempt_result.carry_after_attempt),
        y=tangent_inputs.dy,
        dt=tangent_inputs.dh,
        lagged_reference_y=tangent_inputs.dy,
    )
    return _RadauAcceptedStepAttemptResult(
        carry_after_attempt=carry_after_attempt_tangent,
        trial_dt=tangent_result.dtrial_dt,
        trial_y=tangent_result.dtrial_y,
        err_norm=_zero_tangent_like(attempt_result.err_norm),
        converged=_zero_tangent_like(attempt_result.converged),
        stage_history=tangent_result.dstage_history,
        jacobian_out=_zero_tangent_like(attempt_result.jacobian_out),
        cache_valid_out=_zero_tangent_like(attempt_result.cache_valid_out),
        cache_dt_out=_zero_tangent_like(attempt_result.cache_dt_out),
        cache_age_out=_zero_tangent_like(attempt_result.cache_age_out),
        real_lu_out=_zero_tangent_like(attempt_result.real_lu_out),
        real_piv_out=_zero_tangent_like(attempt_result.real_piv_out),
        complex_lu_out=_zero_tangent_like(attempt_result.complex_lu_out),
        complex_piv_out=_zero_tangent_like(attempt_result.complex_piv_out),
        newton_shrink=_zero_tangent_like(attempt_result.newton_shrink),
        diverged_final=_zero_tangent_like(attempt_result.diverged_final),
        nonfinite_stage_state=_zero_tangent_like(attempt_result.nonfinite_stage_state),
        nonfinite_stage_residual=_zero_tangent_like(attempt_result.nonfinite_stage_residual),
        finite_f0=_zero_tangent_like(attempt_result.finite_f0),
        finite_z0=_zero_tangent_like(attempt_result.finite_z0),
        finite_initial_residual=_zero_tangent_like(attempt_result.finite_initial_residual),
        newton_iter_count=_zero_tangent_like(attempt_result.newton_iter_count),
        final_residual_norm=_zero_tangent_like(attempt_result.final_residual_norm),
        final_delta_norm=_zero_tangent_like(attempt_result.final_delta_norm),
        theta_final=_zero_tangent_like(attempt_result.theta_final),
        slow_contraction_final=_zero_tangent_like(attempt_result.slow_contraction_final),
        residual_blowup_final=_zero_tangent_like(attempt_result.residual_blowup_final),
        newton_nonfinite_final=_zero_tangent_like(attempt_result.newton_nonfinite_final),
        density_err_norm=_zero_tangent_like(attempt_result.density_err_norm),
        pressure_err_norm=_zero_tangent_like(attempt_result.pressure_err_norm),
        er_err_norm=_zero_tangent_like(attempt_result.er_err_norm),
        lagged_response_reused=_zero_tangent_like(attempt_result.lagged_response_reused),
        jacobian_reused=_zero_tangent_like(attempt_result.jacobian_reused),
    )


def _radau_compute_approximate_attempt_tangent(
    kernel_context: _RadauAcceptedStepKernelContext,
    *,
    tangent_inputs: _RadauAcceptedStepTangentInputs,
    jacobian_ref,
    trial_dt,
    stage_history,
    real_lu_out,
    real_piv_out,
    complex_lu_out,
    complex_piv_out,
) -> _RadauAcceptedStepApproximateTangentResult:
    """Compute the first approximate implicit-diff tangent for one accepted step.

    This helper packages the current solver-mathematical approximation into the
    accepted-step output shape we expect a future JVP rule to produce.
    """
    dy_next, dz_stages = _radau_approximate_accepted_step_tangent(
        kernel_context,
        jacobian_ref=jacobian_ref,
        h_value=trial_dt,
        stages_final=stage_history.reshape((kernel_context.num_stages, kernel_context.state_dim)),
        dy_source=tangent_inputs.dy,
        dh_source=tangent_inputs.dh,
        real_lu_out=real_lu_out,
        real_piv_out=real_piv_out,
        complex_lu_out=complex_lu_out,
        complex_piv_out=complex_piv_out,
    )
    return _RadauAcceptedStepApproximateTangentResult(
        dy_next=dy_next,
        dz_stages=dz_stages,
        dtrial_dt=tangent_inputs.dh,
        dtrial_y=dy_next,
        dstage_history=dz_stages.reshape((-1,)),
    )


def _execute_radau_accepted_step_attempt_with_approx_tangent(
    kernel_context: _RadauAcceptedStepKernelContext,
    physics_context: _RadauAcceptedStepPhysicsContext,
    carry_in: _RadauAcceptedStepCarry,
    carry_tangent: _RadauAcceptedStepCarry,
    context: _RadauAcceptedStepAttemptContext,
) -> tuple[_RadauAcceptedStepAttemptResult, _RadauAcceptedStepAttemptResult]:
    """Compute primal accepted-step result plus the first approximate tangent.

    This is not yet wired into JAX as a custom derivative rule. It is a pure
    solver-mathematical helper that defines what a future accepted-step JVP
    should return.
    """
    primal_result = _execute_radau_accepted_step_attempt(
        kernel_context,
        physics_context,
        carry_in,
        context,
    )
    tangent_inputs = _radau_extract_tangent_inputs_from_carry(carry_tangent)
    tangent_result = _radau_compute_approximate_attempt_tangent(
        kernel_context,
        tangent_inputs=tangent_inputs,
        jacobian_ref=primal_result.jacobian_out,
        trial_dt=primal_result.trial_dt,
        stage_history=primal_result.stage_history,
        real_lu_out=primal_result.real_lu_out,
        real_piv_out=primal_result.real_piv_out,
        complex_lu_out=primal_result.complex_lu_out,
        complex_piv_out=primal_result.complex_piv_out,
    )
    tangent_attempt = _radau_build_approximate_tangent_result(
        tangent_inputs,
        tangent_result,
        attempt_result=primal_result,
    )
    return primal_result, tangent_attempt


@partial(jax.custom_jvp, nondiff_argnums=(0, 1, 3))
def _execute_radau_accepted_step_attempt_autodiff(
    kernel_context: _RadauAcceptedStepKernelContext,
    physics_context: _RadauAcceptedStepPhysicsContext,
    carry_in: _RadauAcceptedStepCarry,
    context: _RadauAcceptedStepAttemptContext,
) -> _RadauAcceptedStepAttemptResult:
    """AD-facing accepted-step attempt wrapper using the Radau-native JVP.

    The forward pass remains exactly the current accepted-step primal. The JVP
    avoids differentiating through the raw LU-factorisation / Newton-iteration
    internals and instead uses the existing approximate implicit-diff tangent
    defined at the accepted-step boundary.
    """
    return _execute_radau_accepted_step_attempt(
        kernel_context,
        physics_context,
        carry_in,
        context,
    )


@_execute_radau_accepted_step_attempt_autodiff.defjvp
def _execute_radau_accepted_step_attempt_autodiff_jvp(
    kernel_context: _RadauAcceptedStepKernelContext,
    physics_context: _RadauAcceptedStepPhysicsContext,
    context: _RadauAcceptedStepAttemptContext,
    primals,
    tangents,
):
    (carry_in,) = primals
    (carry_tangent,) = tangents
    return _execute_radau_accepted_step_attempt_with_approx_tangent(
        kernel_context,
        physics_context,
        carry_in,
        carry_tangent,
        context,
    )


def _execute_radau_accepted_step_attempt(
    kernel_context: _RadauAcceptedStepKernelContext,
    physics_context: _RadauAcceptedStepPhysicsContext,
    carry_in: _RadauAcceptedStepCarry,
    context: _RadauAcceptedStepAttemptContext,
) -> _RadauAcceptedStepAttemptResult:
    """Module-scope accepted-step attempt wrapper.

    This preserves the current forward behavior, but moves the accepted-step
    boundary itself out of the nested solver closure so a future custom
    derivative rule can attach at a cleaner level.
    """
    trial_dt = jnp.minimum(carry_in.dt, context.t_final - carry_in.t)
    (
        trial_y, err_norm, converged, stage_history, theta_final,
        newton_iter_count, final_residual_norm, final_delta_norm,
        slow_contraction_final, residual_blowup_final, newton_nonfinite_final,
        jacobian_out, cache_valid_out, cache_dt_out, cache_age_out,
        real_lu_out, real_piv_out, complex_lu_out, complex_piv_out,
        newton_shrink, diverged_final, nonfinite_stage_state, nonfinite_stage_residual,
        finite_f0, finite_z0, finite_initial_residual, density_err_norm, pressure_err_norm, er_err_norm,
        lagged_response_out, lagged_reference_y_out, lagged_response_reused,
        jacobian_reused,
    ) = _radau_single_step_primal(kernel_context, physics_context, carry_in, trial_dt)
    carry_after_attempt = dataclasses.replace(
        carry_in,
        lagged_response_cache=lagged_response_out,
        lagged_response_valid=jnp.asarray(context.use_transport_lagged_response),
        lagged_reference_y=lagged_reference_y_out,
    )
    return _RadauAcceptedStepAttemptResult(
        carry_after_attempt=carry_after_attempt,
        trial_dt=trial_dt,
        trial_y=trial_y,
        err_norm=err_norm,
        converged=converged,
        stage_history=stage_history,
        jacobian_out=jacobian_out,
        cache_valid_out=cache_valid_out,
        cache_dt_out=cache_dt_out,
        cache_age_out=cache_age_out,
        real_lu_out=real_lu_out,
        real_piv_out=real_piv_out,
        complex_lu_out=complex_lu_out,
        complex_piv_out=complex_piv_out,
        newton_shrink=newton_shrink,
        diverged_final=diverged_final,
        nonfinite_stage_state=nonfinite_stage_state,
        nonfinite_stage_residual=nonfinite_stage_residual,
        finite_f0=finite_f0,
        finite_z0=finite_z0,
        finite_initial_residual=finite_initial_residual,
        newton_iter_count=newton_iter_count,
        final_residual_norm=final_residual_norm,
        final_delta_norm=final_delta_norm,
        theta_final=theta_final,
        slow_contraction_final=slow_contraction_final,
        residual_blowup_final=residual_blowup_final,
        newton_nonfinite_final=newton_nonfinite_final,
        density_err_norm=density_err_norm,
        pressure_err_norm=pressure_err_norm,
        er_err_norm=er_err_norm,
        lagged_response_reused=lagged_response_reused,
        jacobian_reused=jacobian_reused,
    )


def _radau_apply_accepted_step_map(
    kernel_context: _RadauAcceptedStepKernelContext,
    physics_context: _RadauAcceptedStepPhysicsContext,
    carry_in: _RadauAcceptedStepCarry,
    attempt_context: _RadauAcceptedStepAttemptContext,
) -> _RadauAcceptedStepMapResult:
    """Apply one accepted-step composition map outside the adaptive loop.

    This keeps the existing accepted-step primal math intact, but exposes the
    accepted-step composition boundary as its own first-class object for future
    AD-facing multi-step composition work.
    """
    attempt_result = _execute_radau_accepted_step_attempt(
        kernel_context,
        physics_context,
        _radau_carry_with_forward_only_jvp_fields(carry_in),
        attempt_context,
    )
    accepted_y = _project_flat_state_if_needed(
        attempt_result.trial_y,
        physics_context.project_flat,
    )
    next_carry = dataclasses.replace(
        attempt_result.carry_after_attempt,
        t=carry_in.t + attempt_result.trial_dt,
        y=accepted_y,
        dt=attempt_result.trial_dt,
        prev_stages=attempt_result.stage_history,
        prev_dt=attempt_result.trial_dt,
        prev_theta_final=attempt_result.theta_final,
        prev_newton_iter_count=attempt_result.newton_iter_count,
        jacobian=attempt_result.jacobian_out,
        cache_valid=attempt_result.cache_valid_out,
        cache_dt=attempt_result.cache_dt_out,
        cache_age=attempt_result.cache_age_out,
        real_lu=attempt_result.real_lu_out,
        real_piv=attempt_result.real_piv_out,
        complex_lu=attempt_result.complex_lu_out,
        complex_piv=attempt_result.complex_piv_out,
    )
    return _RadauAcceptedStepMapResult(
        next_carry=next_carry,
        accepted_y=accepted_y,
        accepted_t=next_carry.t,
        accepted_dt=attempt_result.trial_dt,
        err_norm=attempt_result.err_norm,
        converged=attempt_result.converged,
        stage_history=attempt_result.stage_history,
        theta_final=attempt_result.theta_final,
        newton_iter_count=attempt_result.newton_iter_count,
        final_residual_norm=attempt_result.final_residual_norm,
        final_delta_norm=attempt_result.final_delta_norm,
        density_err_norm=attempt_result.density_err_norm,
        pressure_err_norm=attempt_result.pressure_err_norm,
        er_err_norm=attempt_result.er_err_norm,
    )


def _radau_attempt_step_lean(
    execution_context: _RadauSolveExecutionContext,
    step_state: _RadauStepState,
):
    status = step_state.status
    fail_code = status[1]
    n_accepted = status[2]
    carry_in = _radau_carry_from_step_state(step_state)
    attempt_result = _execute_radau_accepted_step_attempt(
        execution_context.kernel_context,
        execution_context.physics_context,
        _radau_carry_with_forward_only_jvp_fields(carry_in),
        execution_context.attempt_context,
    )
    step_state_attempt = _radau_step_state_from_carry(attempt_result.carry_after_attempt, status=status)
    return _apply_radau_lean_timestep_controller(
        step_state=step_state_attempt,
        trial_dt=attempt_result.trial_dt,
        trial_y=attempt_result.trial_y,
        err_norm=attempt_result.err_norm,
        density_err_norm=attempt_result.density_err_norm,
        pressure_err_norm=attempt_result.pressure_err_norm,
        er_err_norm=attempt_result.er_err_norm,
        converged=attempt_result.converged,
        stage_history=attempt_result.stage_history,
        jacobian_out=attempt_result.jacobian_out,
        cache_valid_out=attempt_result.cache_valid_out,
        cache_dt_out=attempt_result.cache_dt_out,
        cache_age_out=attempt_result.cache_age_out,
        real_lu_out=attempt_result.real_lu_out,
        real_piv_out=attempt_result.real_piv_out,
        complex_lu_out=attempt_result.complex_lu_out,
        complex_piv_out=attempt_result.complex_piv_out,
        newton_shrink=attempt_result.newton_shrink,
        diverged_final=attempt_result.diverged_final,
        nonfinite_stage_state=attempt_result.nonfinite_stage_state,
        nonfinite_stage_residual=attempt_result.nonfinite_stage_residual,
        finite_f0=attempt_result.finite_f0,
        finite_z0=attempt_result.finite_z0,
        finite_initial_residual=attempt_result.finite_initial_residual,
        newton_iter_count=attempt_result.newton_iter_count,
        final_residual_norm=attempt_result.final_residual_norm,
        final_delta_norm=attempt_result.final_delta_norm,
        theta_final=attempt_result.theta_final,
        slow_contraction=attempt_result.slow_contraction_final,
        residual_blowup=attempt_result.residual_blowup_final,
        newton_nonfinite=attempt_result.newton_nonfinite_final,
        lagged_reused=attempt_result.lagged_response_reused,
        jacobian_reused=attempt_result.jacobian_reused,
        fail_code=fail_code,
        n_accepted=n_accepted,
        dtype=execution_context.dtype,
        dt_min=execution_context.dt_min,
        dt_max=execution_context.dt_max,
        safety_factor=execution_context.safety_factor,
        controller_alpha=execution_context.controller_alpha,
        min_step_factor=execution_context.min_step_factor,
        max_step_factor=execution_context.max_step_factor,
        controller_mode=execution_context.controller_mode,
        use_transport_lagged_response=execution_context.use_transport_lagged_response,
        lagged_response_reuse_mode=execution_context.lagged_response_reuse_mode,
        lagged_response_reuse_rtol=execution_context.lagged_response_reuse_rtol,
        lagged_response_reuse_atol=execution_context.lagged_response_reuse_atol,
        project_flat=execution_context.project_flat,
    )


def _radau_attempt_step_with_payload(
    execution_context: _RadauSolveExecutionContext,
    step_state: _RadauStepState,
):
    carry_in = _radau_carry_from_step_state(step_state)
    attempt_result = _execute_radau_accepted_step_attempt(
        execution_context.kernel_context,
        execution_context.physics_context,
        _radau_carry_with_forward_only_jvp_fields(carry_in),
        execution_context.attempt_context,
    )
    step_state_attempt = _radau_step_state_from_carry(attempt_result.carry_after_attempt, status=step_state.status)
    next_step_state, step_info = _apply_radau_lean_timestep_controller(
        step_state=step_state_attempt,
        trial_dt=attempt_result.trial_dt,
        trial_y=attempt_result.trial_y,
        err_norm=attempt_result.err_norm,
        density_err_norm=attempt_result.density_err_norm,
        pressure_err_norm=attempt_result.pressure_err_norm,
        er_err_norm=attempt_result.er_err_norm,
        converged=attempt_result.converged,
        stage_history=attempt_result.stage_history,
        jacobian_out=attempt_result.jacobian_out,
        cache_valid_out=attempt_result.cache_valid_out,
        cache_dt_out=attempt_result.cache_dt_out,
        cache_age_out=attempt_result.cache_age_out,
        real_lu_out=attempt_result.real_lu_out,
        real_piv_out=attempt_result.real_piv_out,
        complex_lu_out=attempt_result.complex_lu_out,
        complex_piv_out=attempt_result.complex_piv_out,
        newton_shrink=attempt_result.newton_shrink,
        diverged_final=attempt_result.diverged_final,
        nonfinite_stage_state=attempt_result.nonfinite_stage_state,
        nonfinite_stage_residual=attempt_result.nonfinite_stage_residual,
        finite_f0=attempt_result.finite_f0,
        finite_z0=attempt_result.finite_z0,
        finite_initial_residual=attempt_result.finite_initial_residual,
        newton_iter_count=attempt_result.newton_iter_count,
        final_residual_norm=attempt_result.final_residual_norm,
        final_delta_norm=attempt_result.final_delta_norm,
        theta_final=attempt_result.theta_final,
        slow_contraction=attempt_result.slow_contraction_final,
        residual_blowup=attempt_result.residual_blowup_final,
        newton_nonfinite=attempt_result.newton_nonfinite_final,
        lagged_reused=attempt_result.lagged_response_reused,
        jacobian_reused=attempt_result.jacobian_reused,
        fail_code=step_state.status[1],
        n_accepted=step_state.status[2],
        dtype=execution_context.dtype,
        dt_min=execution_context.dt_min,
        dt_max=execution_context.dt_max,
        safety_factor=execution_context.safety_factor,
        controller_alpha=execution_context.controller_alpha,
        min_step_factor=execution_context.min_step_factor,
        max_step_factor=execution_context.max_step_factor,
        controller_mode=execution_context.controller_mode,
        use_transport_lagged_response=execution_context.use_transport_lagged_response,
        lagged_response_reuse_mode=execution_context.lagged_response_reuse_mode,
        lagged_response_reuse_rtol=execution_context.lagged_response_reuse_rtol,
        lagged_response_reuse_atol=execution_context.lagged_response_reuse_atol,
        project_flat=execution_context.project_flat,
    )
    payload = _radau_backward_payload_candidate(carry_in, attempt_result)
    return next_step_state, step_info, payload


def _radau_step_fn(
    execution_context: _RadauSolveExecutionContext,
    step_state: _RadauStepState,
    _,
):
    failed = step_state.status[0] != 0
    fail_code = step_state.status[1]

    def _skip(_):
        return step_state, _RadauStepInfo(
            y=step_state.y,
            t=step_state.t,
            dt=jnp.asarray(0.0, dtype=execution_context.dtype),
            next_dt=step_state.dt,
            growth=jnp.asarray(1.0, dtype=execution_context.dtype),
            lagged_reused=jnp.asarray(False),
            jacobian_reused=jnp.asarray(False),
            accepted=jnp.asarray(False),
            failed=failed,
            fail_code=fail_code,
            converged=jnp.asarray(False),
            err_norm=jnp.asarray(jnp.inf, dtype=execution_context.dtype),
            diverged=jnp.asarray(False),
            nonfinite_stage_state=jnp.asarray(False),
            nonfinite_stage_residual=jnp.asarray(False),
            finite_f0=jnp.asarray(True),
            finite_z0=jnp.asarray(True),
            finite_initial_residual=jnp.asarray(True),
            newton_iter_count=jnp.asarray(0, dtype=jnp.int32),
            final_residual_norm=jnp.asarray(jnp.inf, dtype=execution_context.dtype),
            final_delta_norm=jnp.asarray(jnp.inf, dtype=execution_context.dtype),
            theta_final=jnp.asarray(0.0, dtype=execution_context.dtype),
            slow_contraction=jnp.asarray(False),
            residual_blowup=jnp.asarray(False),
            newton_nonfinite=jnp.asarray(False),
        )

    def _run(_):
        return _radau_attempt_step_lean(execution_context, step_state)

    step_state_out, step_info = jax.lax.cond(failed, _skip, _run, operand=None)
    if execution_context.debug_newton_trace:
        jax.debug.print(
            "[radau-solver] attempt t_start={t_start:.6e} dt_try={dt_try:.6e} accepted={accepted} failed={failed} fail_code={fail_code} converged={converged} err_norm={err_norm:.6e} growth={growth:.6e} next_dt={next_dt:.6e} lagged_reused={lagged_reused} jacobian_reused={jacobian_reused}",
            t_start=step_state.t,
            dt_try=step_state.dt,
            accepted=step_info.accepted,
            failed=step_info.failed,
            fail_code=step_info.fail_code,
            converged=step_info.converged,
            err_norm=jnp.asarray(jnp.inf, dtype=execution_context.dtype) if getattr(step_info, "err_norm", None) is None else jnp.asarray(getattr(step_info, "err_norm"), dtype=execution_context.dtype),
            growth=jnp.asarray(1.0, dtype=execution_context.dtype) if getattr(step_info, "growth", None) is None else jnp.asarray(getattr(step_info, "growth"), dtype=execution_context.dtype),
            next_dt=step_state.dt if getattr(step_info, "next_dt", None) is None else jnp.asarray(getattr(step_info, "next_dt"), dtype=execution_context.dtype),
            lagged_reused=jnp.asarray(False) if getattr(step_info, "lagged_reused", None) is None else jnp.asarray(getattr(step_info, "lagged_reused")),
            jacobian_reused=jnp.asarray(False) if getattr(step_info, "jacobian_reused", None) is None else jnp.asarray(getattr(step_info, "jacobian_reused")),
            ordered=True,
        )
    return step_state_out, step_info


def _radau_prepare_lagged_response(
    kernel_context: _RadauAcceptedStepKernelContext,
    carry_in: _RadauAcceptedStepCarry,
    unpack_flat: Callable[[Any], Any],
    project_flat: Callable[[Any], Any] | None,
    build_lagged_response: Callable[[Any], Any] | None,
) -> tuple[Any, Any, Any]:
    """Prepare optional lagged-response data for one accepted-step attempt."""
    lagged_response_reused = jnp.asarray(False)
    if kernel_context.use_transport_lagged_response:
        flat_y = carry_in.y
        candidate_state = unpack_flat(_project_flat_state_if_needed(flat_y, project_flat))
        lagged_response_reused = jnp.asarray(carry_in.lagged_response_valid)

        def _reuse_cached(_):
            return carry_in.lagged_response_cache

        def _rebuild_cached(_):
            if build_lagged_response is None:
                return None
            return build_lagged_response(candidate_state)

        lagged_response = jax.lax.cond(
            carry_in.lagged_response_valid,
            _reuse_cached,
            _rebuild_cached,
            operand=None,
        )
        lagged_reference_y = jax.lax.cond(
            carry_in.lagged_response_valid,
            lambda _: carry_in.lagged_reference_y,
            lambda _: flat_y,
            operand=None,
        )
        return lagged_response, lagged_reference_y, lagged_response_reused
    return None, carry_in.y, lagged_response_reused


def _radau_eval_rhs(
    t_eval,
    y_eval,
    lagged_response,
    flat_rhs: Callable[[Any, Any], Any],
    flat_rhs_with_lagged_response: Callable[[Any, Any, Any], Any],
):
    """Evaluate RHS with optional lagged-response data."""
    if lagged_response is not None:
        return flat_rhs_with_lagged_response(t_eval, y_eval, lagged_response)
    return flat_rhs(t_eval, y_eval)


def _radau_residual_norm(
    kernel_context: _RadauAcceptedStepKernelContext,
    residual_vec,
):
    raw_norm = jnp.linalg.norm(residual_vec)
    return jnp.where(
        kernel_context.use_rms_residual_norm,
        raw_norm / jnp.maximum(kernel_context.residual_size_sqrt, jnp.asarray(1.0, dtype=kernel_context.dtype)),
        raw_norm,
    )


def _radau_correction_norm(
    kernel_context: _RadauAcceptedStepKernelContext,
    delta_vec,
):
    raw_norm = jnp.linalg.norm(delta_vec)
    scaled_norm = jnp.sqrt(
        jnp.mean((delta_vec / kernel_context.stage_scale) * (delta_vec / kernel_context.stage_scale))
        + kernel_context.tiny_scalar
    )
    return jnp.where(kernel_context.use_hairer_scaled_correction, scaled_norm, raw_norm)


def _radau_transform_stage_stack(
    kernel_context: _RadauAcceptedStepKernelContext,
    stage_stack,
):
    return kernel_context.radau_inv_transform @ stage_stack


def _radau_inverse_transform_stage_stack(
    kernel_context: _RadauAcceptedStepKernelContext,
    stage_stack,
):
    return kernel_context.radau_transform @ stage_stack


def _radau_evaluate_stage_model(
    kernel_context: _RadauAcceptedStepKernelContext,
    physics_context: _RadauAcceptedStepPhysicsContext,
    *,
    flat_y,
    t_value,
    h_value,
    z_flat,
    f0,
    jacobian_ref,
    lagged_response,
):
    stages = z_flat.reshape((kernel_context.num_stages, kernel_context.state_dim))
    stage_states = flat_y[None, :] + h_value * (kernel_context.a @ stages)
    if lagged_response is not None:
        stage_times = t_value + kernel_context.c * h_value
        evals = jax.vmap(
            lambda t_eval, y_eval: _radau_eval_rhs(
                t_eval,
                y_eval,
                lagged_response,
                physics_context.flat_rhs,
                physics_context.flat_rhs_with_lagged_response,
            ),
            in_axes=(0, 0),
        )(stage_times, stage_states)
    elif kernel_context.use_lagged_linear_response:
        state_delta = stage_states - flat_y[None, :]
        evals = f0[None, :] + state_delta @ jacobian_ref.T
    else:
        stage_times = t_value + kernel_context.c * h_value
        evals = jax.vmap(physics_context.flat_rhs, in_axes=(0, 0))(stage_times, stage_states)
    return stages, evals


def _radau_stage_residual(
    kernel_context: _RadauAcceptedStepKernelContext,
    physics_context: _RadauAcceptedStepPhysicsContext,
    *,
    flat_y,
    t_value,
    h_value,
    z_flat,
    f0,
    jacobian_ref,
    lagged_response,
):
    stages, evals = _radau_evaluate_stage_model(
        kernel_context,
        physics_context,
        flat_y=flat_y,
        t_value=t_value,
        h_value=h_value,
        z_flat=z_flat,
        f0=f0,
        jacobian_ref=jacobian_ref,
        lagged_response=lagged_response,
    )
    return (stages - evals).reshape((-1,))


def _radau_apply_stage_linear_solve(
    kernel_context: _RadauAcceptedStepKernelContext,
    *,
    rhs,
    real_lu_out,
    real_piv_out,
    complex_lu_out,
    complex_piv_out,
):
    rhs_arr = jnp.asarray(rhs, dtype=kernel_context.dtype).reshape((-1,))
    zero_rhs = jnp.all(rhs_arr == jnp.asarray(0.0, dtype=kernel_context.dtype))

    def _solve_zero_rhs(rhs_flat):
        return jnp.zeros_like(rhs_flat)

    def _solve_nonzero_rhs(rhs_flat):
        rhs_stages = rhs_flat.reshape((kernel_context.num_stages, kernel_context.state_dim))
        rhs_transformed = _radau_transform_stage_stack(kernel_context, rhs_stages)
        rhs_real = rhs_transformed[0]
        delta_real = jax.scipy.linalg.lu_solve((real_lu_out, real_piv_out), rhs_real)
        rhs_complex_pairs = rhs_transformed[1:].reshape((kernel_context.num_complex_pairs, 2, kernel_context.state_dim))

        def _solve_pair(i, pair_solutions):
            delta_pair = jax.scipy.linalg.lu_solve(
                (complex_lu_out[i], complex_piv_out[i]),
                rhs_complex_pairs[i].reshape((-1,)),
            ).reshape((2, kernel_context.state_dim))
            return pair_solutions.at[i].set(delta_pair)

        delta_complex_pairs = jax.lax.fori_loop(
            0,
            kernel_context.num_complex_pairs,
            _solve_pair,
            jnp.zeros_like(rhs_complex_pairs),
        )
        delta_transformed = jnp.concatenate(
            [delta_real[None, :], delta_complex_pairs.reshape((2 * kernel_context.num_complex_pairs, kernel_context.state_dim))],
            axis=0,
        )
        return _radau_inverse_transform_stage_stack(kernel_context, delta_transformed).reshape((-1,))

    return jax.lax.cond(zero_rhs, _solve_zero_rhs, _solve_nonzero_rhs, rhs_arr)


def _radau_approximate_accepted_step_tangent(
    kernel_context: _RadauAcceptedStepKernelContext,
    *,
    jacobian_ref,
    h_value,
    stages_final,
    dy_source,
    dh_source,
    real_lu_out,
    real_piv_out,
    complex_lu_out,
    complex_piv_out,
):
    """Approximate accepted-step tangent via the current Newton linearization.

    This helper uses the same stage-space linear operator already built for the
    Newton solve. It is the first Radau-native tangent building block:

    - it differentiates the converged accepted-step equations
    - it does not differentiate through the raw Newton iteration history
    - it currently uses the existing Jacobian linearization rather than the
      exact collocation Jacobian at the converged stage point

    The linearized stage system used here is:

    (I - h A ⊗ J_ref) dZ = J_ref (dy + dh * A Z)

    and the accepted-state tangent is:

    dy_next = dy + dh * (b^T Z) + h * (b^T dZ)
    """
    dy_source = jnp.asarray(dy_source, dtype=kernel_context.dtype)
    dh_source = jnp.asarray(dh_source, dtype=kernel_context.dtype)
    zero_dy = jnp.all(dy_source == jnp.asarray(0.0, dtype=kernel_context.dtype))
    zero_dh = dh_source == jnp.asarray(0.0, dtype=kernel_context.dtype)
    zero_input = jnp.logical_and(zero_dy, zero_dh)

    def _zero_tangent(_):
        dz_zero = jnp.zeros((kernel_context.num_stages, kernel_context.state_dim), dtype=kernel_context.dtype)
        dy_zero = jnp.zeros((kernel_context.state_dim,), dtype=kernel_context.dtype)
        return dy_zero, dz_zero

    def _compute_tangent(_):
        stage_state_source = dy_source[None, :] + dh_source * (kernel_context.a @ stages_final)
        stage_rhs = stage_state_source @ jacobian_ref.T
        dz_flat = _radau_apply_stage_linear_solve(
            kernel_context,
            rhs=stage_rhs.reshape((-1,)),
            real_lu_out=real_lu_out,
            real_piv_out=real_piv_out,
            complex_lu_out=complex_lu_out,
            complex_piv_out=complex_piv_out,
        )
        dz_stages = dz_flat.reshape((kernel_context.num_stages, kernel_context.state_dim))
        dy_next = (
            dy_source
            + dh_source * (kernel_context.b @ stages_final)
            + h_value * (kernel_context.b @ dz_stages)
        )
        return dy_next, dz_stages

    return jax.lax.cond(zero_input, _zero_tangent, _compute_tangent, operand=None)


def _radau_carry_with_forward_only_jvp_fields(
    carry: _RadauAcceptedStepCarry,
) -> _RadauAcceptedStepCarry:
    """Mask clearly forward-only or recomputable carry fields in the JVP path.

    This is the first conservative reduction of the differentiated object:
    controller bookkeeping and Jacobian/LU reuse state remain active in the
    forward solve, but do not contribute tangents through the accepted-step
    JVP boundary.
    """
    return dataclasses.replace(
        carry,
        prev_error=jax.lax.stop_gradient(carry.prev_error),
        recent_reject_count=jax.lax.stop_gradient(carry.recent_reject_count),
        regrowth_cooldown=jax.lax.stop_gradient(carry.regrowth_cooldown),
        easy_growth_streak=jax.lax.stop_gradient(carry.easy_growth_streak),
        jacobian=jax.lax.stop_gradient(carry.jacobian),
        cache_valid=jax.lax.stop_gradient(carry.cache_valid),
        cache_dt=jax.lax.stop_gradient(carry.cache_dt),
        cache_age=jax.lax.stop_gradient(carry.cache_age),
        real_lu=jax.lax.stop_gradient(carry.real_lu),
        real_piv=jax.lax.stop_gradient(carry.real_piv),
        complex_lu=jax.lax.stop_gradient(carry.complex_lu),
        complex_piv=jax.lax.stop_gradient(carry.complex_piv),
    )


def _radau_step_state_with_forward_only_controller_fields(
    step_state: _RadauStepState,
) -> _RadauStepState:
    """Mask controller-evolution fields while keeping state evolution active."""
    return dataclasses.replace(
        step_state,
        dt=jax.lax.stop_gradient(step_state.dt),
        status=jax.lax.stop_gradient(step_state.status),
        prev_error=jax.lax.stop_gradient(step_state.prev_error),
        prev_dt=jax.lax.stop_gradient(step_state.prev_dt),
        recent_reject_count=jax.lax.stop_gradient(step_state.recent_reject_count),
        regrowth_cooldown=jax.lax.stop_gradient(step_state.regrowth_cooldown),
        easy_growth_streak=jax.lax.stop_gradient(step_state.easy_growth_streak),
        jacobian=jax.lax.stop_gradient(step_state.jacobian),
        cache_valid=jax.lax.stop_gradient(step_state.cache_valid),
        cache_dt=jax.lax.stop_gradient(step_state.cache_dt),
        cache_age=jax.lax.stop_gradient(step_state.cache_age),
        real_lu=jax.lax.stop_gradient(step_state.real_lu),
        real_piv=jax.lax.stop_gradient(step_state.real_piv),
        complex_lu=jax.lax.stop_gradient(step_state.complex_lu),
        complex_piv=jax.lax.stop_gradient(step_state.complex_piv),
        prev_theta_final=jax.lax.stop_gradient(step_state.prev_theta_final),
        prev_newton_iter_count=jax.lax.stop_gradient(step_state.prev_newton_iter_count),
    )


def _radau_stage_subsolve_residual(
    kernel_context: _RadauAcceptedStepKernelContext,
    physics_context: _RadauAcceptedStepPhysicsContext,
    inputs: _RadauStageSubsolveInputs,
    z_flat,
):
    """Evaluate the stage residual from explicit subsolve inputs."""
    return _radau_stage_residual(
        kernel_context,
        physics_context,
        flat_y=inputs.flat_y,
        t_value=inputs.t_value,
        h_value=inputs.h_value,
        z_flat=z_flat,
        f0=inputs.f0,
        jacobian_ref=inputs.jacobian_ref,
        lagged_response=inputs.lagged_response,
    )


def _radau_stage_subsolve_linear_solve(
    kernel_context: _RadauAcceptedStepKernelContext,
    inputs: _RadauStageSubsolveInputs,
    rhs,
):
    """Apply the current Newton linear solve from explicit subsolve inputs."""
    return _radau_apply_stage_linear_solve(
        kernel_context,
        rhs=rhs,
        real_lu_out=inputs.real_lu_out,
        real_piv_out=inputs.real_piv_out,
        complex_lu_out=inputs.complex_lu_out,
        complex_piv_out=inputs.complex_piv_out,
    )


def _radau_prepare_stage_subsolve_inputs_from_carry(
    kernel_context: _RadauAcceptedStepKernelContext,
    physics_context: _RadauAcceptedStepPhysicsContext,
    carry_in: _RadauAcceptedStepCarry,
    *,
    t_final,
) -> _RadauStageSubsolveInputs:
    """Prepare explicit stage-subsolve inputs from a real Radau carry.

    This is the standalone validation bridge between the production Radau setup
    and the AD-facing stage-subsolve primitive.
    """
    flat_y = carry_in.y
    t_value = carry_in.t
    prev_stages = carry_in.prev_stages
    prev_dt = carry_in.prev_dt
    prev_theta_final = carry_in.prev_theta_final
    prev_newton_iter_count = carry_in.prev_newton_iter_count
    jacobian_cache = carry_in.jacobian
    cache_valid = carry_in.cache_valid
    cache_dt = carry_in.cache_dt
    real_lu_cache = carry_in.real_lu
    real_piv_cache = carry_in.real_piv
    complex_lu_cache = carry_in.complex_lu
    complex_piv_cache = carry_in.complex_piv
    lagged_response, _lagged_reference_y, _lagged_response_reused = _radau_prepare_lagged_response(
        kernel_context,
        carry_in,
        physics_context.unpack_flat,
        physics_context.project_flat,
        physics_context.build_lagged_response,
    )
    trial_dt = jnp.minimum(carry_in.dt, jnp.asarray(t_final, dtype=kernel_context.dtype) - t_value)

    def _rhs_eval(t_eval, y_eval):
        return _radau_eval_rhs(
            t_eval,
            y_eval,
            lagged_response,
            physics_context.flat_rhs,
            physics_context.flat_rhs_with_lagged_response,
        )

    def _rhs_eval_at_current_time(y_eval):
        return _rhs_eval(t_value, y_eval)

    f0 = _rhs_eval(t_value, flat_y)
    z0 = _make_radau_stage_predictor(
        f0,
        prev_stages,
        prev_dt,
        trial_dt,
        kernel_context.c,
        kernel_context.dtype,
        density_size=kernel_context.density_size,
        pressure_size=kernel_context.pressure_size,
        er_size=kernel_context.er_size,
        prev_theta_final=prev_theta_final,
        prev_newton_iter_count=prev_newton_iter_count,
        predictor_mode=kernel_context.predictor_mode,
    )
    jacobian_dt_scale = jnp.maximum(
        jnp.abs(cache_dt),
        jnp.asarray(1.0e-14, dtype=kernel_context.dtype),
    )
    dt_close = jnp.abs(trial_dt - cache_dt) <= kernel_context.jacobian_reuse_rtol * jacobian_dt_scale
    reuse_linearization = jnp.logical_and(
        jnp.logical_and(cache_valid, dt_close),
        jnp.logical_not(kernel_context.use_lagged_linear_response),
    )

    def _reuse_linearization(_):
        return jacobian_cache, real_lu_cache, real_piv_cache, complex_lu_cache, complex_piv_cache

    def _recompute_linearization(_):
        jacobian_ref = jax.jacfwd(_rhs_eval_at_current_time)(flat_y)
        h_jacobian = trial_dt * jacobian_ref
        real_matrix = kernel_context.identity_n - kernel_context.radau_real_eig * h_jacobian
        real_lu, real_piv = jax.scipy.linalg.lu_factor(real_matrix)
        complex_dense_all = jnp.transpose(
            kernel_context.identity_2[None, :, :, None, None] * kernel_context.identity_n[None, None, None, :, :]
            - kernel_context.radau_complex_blocks[:, :, :, None, None] * h_jacobian[None, None, None, :, :],
            (0, 1, 3, 2, 4),
        ).reshape((kernel_context.num_complex_pairs, kernel_context.complex_dim, kernel_context.complex_dim))

        def _factor_pair(i, carry):
            lu_all, piv_all = carry
            lu_i, piv_i = jax.scipy.linalg.lu_factor(complex_dense_all[i])
            lu_all = lu_all.at[i].set(lu_i)
            piv_all = piv_all.at[i].set(piv_i)
            return lu_all, piv_all

        complex_lu, complex_piv = jax.lax.fori_loop(
            0,
            kernel_context.num_complex_pairs,
            _factor_pair,
            (jnp.zeros_like(complex_lu_cache), jnp.zeros_like(complex_piv_cache)),
        )
        return jacobian_ref, real_lu, real_piv, complex_lu, complex_piv

    jacobian_ref, real_lu_out, real_piv_out, complex_lu_out, complex_piv_out = jax.lax.cond(
        reuse_linearization,
        _reuse_linearization,
        _recompute_linearization,
        operand=None,
    )
    return _radau_build_stage_subsolve_inputs(
        flat_y=flat_y,
        t_value=t_value,
        h_value=trial_dt,
        z0=z0,
        f0=f0,
        jacobian_ref=jacobian_ref,
        lagged_response=lagged_response,
        real_lu_out=real_lu_out,
        real_piv_out=real_piv_out,
        complex_lu_out=complex_lu_out,
        complex_piv_out=complex_piv_out,
    )


def _radau_build_stage_subsolve_inputs(
    *,
    flat_y,
    t_value,
    h_value,
    z0,
    f0,
    jacobian_ref,
    lagged_response,
    real_lu_out,
    real_piv_out,
    complex_lu_out,
    complex_piv_out,
) -> _RadauStageSubsolveInputs:
    """Package the explicit inputs for one Radau stage subsolve."""
    return _RadauStageSubsolveInputs(
        flat_y=flat_y,
        t_value=t_value,
        h_value=h_value,
        z0=z0,
        f0=f0,
        jacobian_ref=jacobian_ref,
        lagged_response=lagged_response,
        real_lu_out=real_lu_out,
        real_piv_out=real_piv_out,
        complex_lu_out=complex_lu_out,
        complex_piv_out=complex_piv_out,
    )


def _radau_extract_stage_subsolve_tangent_inputs(
    inputs_tangent: _RadauStageSubsolveInputs,
) -> _RadauStageSubsolveTangentInputs:
    """Extract the first tangent-active Radau stage-subsolve inputs.

    The first implicit-diff subsolve rule is only active with respect to:

    - the incoming accepted state `y_n`
    - the trial step size `h`

    while Jacobian/LU factors and lagged-response objects remain forward data.
    """
    return _RadauStageSubsolveTangentInputs(
        dy=inputs_tangent.flat_y,
        dh=inputs_tangent.h_value,
    )


def _radau_compute_stage_subsolve_approximate_tangent(
    kernel_context: _RadauAcceptedStepKernelContext,
    inputs: _RadauStageSubsolveInputs,
    tangent_inputs: _RadauStageSubsolveTangentInputs,
    subsolve_result: _RadauStageSubsolveResult,
) -> _RadauStageSubsolveApproximateTangentResult:
    """Compute the first approximate tangent for the explicit stage subsolve."""
    stages_final = subsolve_result.z_final.reshape((kernel_context.num_stages, kernel_context.state_dim))
    _, dz_stages = _radau_approximate_accepted_step_tangent(
        kernel_context,
        jacobian_ref=inputs.jacobian_ref,
        h_value=inputs.h_value,
        stages_final=stages_final,
        dy_source=tangent_inputs.dy,
        dh_source=tangent_inputs.dh,
        real_lu_out=inputs.real_lu_out,
        real_piv_out=inputs.real_piv_out,
        complex_lu_out=inputs.complex_lu_out,
        complex_piv_out=inputs.complex_piv_out,
    )
    return _RadauStageSubsolveApproximateTangentResult(
        dz_stages=dz_stages,
        dz_flat=dz_stages.reshape((-1,)),
    )


def _radau_build_stage_subsolve_tangent_result(
    tangent_result: _RadauStageSubsolveApproximateTangentResult,
    *,
    primal_result: _RadauStageSubsolveResult,
) -> _RadauStageSubsolveResult:
    """Lift the approximate subsolve tangent back to subsolve-result shape."""

    def _zero_tangent_like(x):
        arr = jnp.asarray(x)
        if jnp.issubdtype(arr.dtype, jnp.inexact):
            return jnp.zeros_like(arr)
        return jnp.zeros(arr.shape, dtype=jax.dtypes.float0)

    return _RadauStageSubsolveResult(
        iter_final=_zero_tangent_like(primal_result.iter_final),
        z_final=tangent_result.dz_flat,
        delta_norm_final=_zero_tangent_like(primal_result.delta_norm_final),
        newton_metric_final=_zero_tangent_like(primal_result.newton_metric_final),
        theta_final=_zero_tangent_like(primal_result.theta_final),
        diverged_final=_zero_tangent_like(primal_result.diverged_final),
        shrink_suggest_final=_zero_tangent_like(primal_result.shrink_suggest_final),
        slow_contraction_final=_zero_tangent_like(primal_result.slow_contraction_final),
        residual_blowup_final=_zero_tangent_like(primal_result.residual_blowup_final),
        newton_nonfinite_final=_zero_tangent_like(primal_result.newton_nonfinite_final),
        finite_initial_residual=_zero_tangent_like(primal_result.finite_initial_residual),
        nonfinite_stage_state=_zero_tangent_like(primal_result.nonfinite_stage_state),
        nonfinite_stage_residual=_zero_tangent_like(primal_result.nonfinite_stage_residual),
        final_residual=_zero_tangent_like(primal_result.final_residual),
        final_residual_norm=_zero_tangent_like(primal_result.final_residual_norm),
        converged=_zero_tangent_like(primal_result.converged),
    )


def _radau_run_stage_subsolve_with_approx_tangent(
    kernel_context: _RadauAcceptedStepKernelContext,
    physics_context: _RadauAcceptedStepPhysicsContext,
    inputs: _RadauStageSubsolveInputs,
    inputs_tangent: _RadauStageSubsolveInputs,
) -> tuple[_RadauStageSubsolveResult, _RadauStageSubsolveApproximateTangentResult]:
    """Return the stage-subsolve primal result plus its first approximate tangent."""
    primal_result = _radau_run_stage_subsolve_from_inputs(
        kernel_context,
        physics_context,
        inputs,
    )
    tangent_inputs = _radau_extract_stage_subsolve_tangent_inputs(inputs_tangent)
    tangent_result = _radau_compute_stage_subsolve_approximate_tangent(
        kernel_context,
        inputs,
        tangent_inputs,
        primal_result,
    )
    return primal_result, tangent_result


@partial(jax.custom_jvp, nondiff_argnums=(0, 1))
def _radau_run_stage_subsolve_standalone_autodiff(
    kernel_context: _RadauAcceptedStepKernelContext,
    physics_context: _RadauAcceptedStepPhysicsContext,
    inputs: _RadauStageSubsolveInputs,
) -> _RadauStageSubsolveResult:
    """Standalone AD-facing Radau stage-subsolve primitive.

    This wrapper is intentionally *not* used by the production adaptive solve
    path. It exists so we can validate the Radau-native custom derivative on
    the mathematically meaningful stage subsolve outside the current
    `solve -> jit(step_fn) -> lax.cond(...)` machinery.
    """
    return _radau_run_stage_subsolve(
        kernel_context,
        physics_context,
        inputs,
    )


@_radau_run_stage_subsolve_standalone_autodiff.defjvp
def _radau_run_stage_subsolve_standalone_autodiff_jvp(
    kernel_context: _RadauAcceptedStepKernelContext,
    physics_context: _RadauAcceptedStepPhysicsContext,
    primals,
    tangents,
):
    (inputs,) = primals
    (inputs_tangent,) = tangents
    primal_result, approx_tangent = _radau_run_stage_subsolve_with_approx_tangent(
        kernel_context,
        physics_context,
        inputs,
        inputs_tangent,
    )
    tangent_result = _radau_build_stage_subsolve_tangent_result(
        approx_tangent,
        primal_result=primal_result,
    )
    return primal_result, tangent_result


def _radau_run_stage_subsolve_from_inputs(
    kernel_context: _RadauAcceptedStepKernelContext,
    physics_context: _RadauAcceptedStepPhysicsContext,
    inputs: _RadauStageSubsolveInputs,
) -> _RadauStageSubsolveResult:
    """Run the stage subsolve from an explicit input bundle.

    This gives the future custom derivative rule a cleaner, closure-light
    primitive target than the full accepted-step or adaptive-loop wrapper.
    """
    return _radau_run_stage_subsolve(
        kernel_context,
        physics_context,
        inputs,
    )


def _radau_run_stage_subsolve(
    kernel_context: _RadauAcceptedStepKernelContext,
    physics_context: _RadauAcceptedStepPhysicsContext,
    inputs: _RadauStageSubsolveInputs,
) -> _RadauStageSubsolveResult:
    """Run the implicit Radau stage Newton solve as a standalone subsolve.

    This isolates the converged stage-system solve from the larger accepted-step
    wrapper and adaptive controller logic. It is the closest NEOPAX analogue to
    SPECTRAX-GK's "customize AD at a meaningful subsolve" pattern.
    """

    def body_fn(newton_state):
        (
            iter_idx,
            z_cur,
            delta_norm,
            residual_norm,
            prev_newton_norm,
            newton_metric,
            prev_theta_ratio,
            theta_est,
            diverged,
            shrink_suggest,
            slow_contraction_any,
            residual_blowup_any,
            newton_nonfinite_any,
        ) = newton_state
        residual_cur = _radau_stage_subsolve_residual(
            kernel_context,
            physics_context,
            inputs,
            z_cur,
        )
        delta = _radau_stage_subsolve_linear_solve(
            kernel_context,
            inputs,
            -residual_cur,
        )
        delta = jnp.where(jnp.all(jnp.isfinite(delta)), delta, jnp.zeros_like(delta))
        z_next = z_cur + delta
        current_residual_norm = _radau_residual_norm(kernel_context, residual_cur)
        current_delta_norm = jnp.linalg.norm(delta)
        current_newton_norm = _radau_correction_norm(kernel_context, delta)
        safe_prev_delta = jnp.maximum(prev_newton_norm, kernel_context.tiny_scalar)
        theta_raw = current_newton_norm / safe_prev_delta
        newton_iter_num = iter_idx + jnp.asarray(1, dtype=jnp.int32)
        theta_candidate = jnp.where(
            newton_iter_num == 2,
            theta_raw,
            jnp.sqrt(jnp.maximum(theta_raw * prev_theta_ratio, kernel_context.tiny_scalar)),
        )
        theta_valid = newton_iter_num > 1
        theta_candidate = jnp.where(theta_valid, theta_candidate, kernel_context.zero_scalar)
        theta_next = jnp.where(theta_valid, theta_candidate, theta_est)
        theta_ratio_next = jnp.where(theta_valid, theta_raw, prev_theta_ratio)
        residual_blowup = jnp.logical_and(
            iter_idx >= 1,
            current_residual_norm > residual_norm * kernel_context.residual_blowup_factor,
        )
        nonfinite_state = jnp.logical_not(
            jnp.logical_and(
                jnp.logical_and(jnp.all(jnp.isfinite(delta)), jnp.isfinite(current_residual_norm)),
                jnp.logical_and(jnp.isfinite(current_delta_norm), jnp.isfinite(current_newton_norm)),
            )
        )
        predictor_active = jnp.logical_and(theta_valid, newton_iter_num < kernel_context.maxiter)
        remaining_iters = jnp.maximum(kernel_context.maxiter - 1 - newton_iter_num, jnp.asarray(0, dtype=jnp.int32))
        faccon = theta_candidate / jnp.maximum(
            jnp.asarray(1.0, dtype=kernel_context.dtype) - theta_candidate,
            kernel_context.tiny_scalar,
        )
        predicted_defect = faccon * current_newton_norm * (theta_candidate ** remaining_iters) / kernel_context.predictor_fnewt
        qnewt = jnp.clip(predicted_defect, kernel_context.predictor_defect_floor, kernel_context.predictor_defect_cap)
        predictor_exponent = -jnp.asarray(1.0, dtype=kernel_context.dtype) / (
            jnp.asarray(kernel_context.maxiter + 3, dtype=kernel_context.dtype) - newton_iter_num.astype(kernel_context.dtype)
        )
        predictor_shrink = jnp.clip(
            kernel_context.newton_shrink_num * (qnewt ** predictor_exponent),
            kernel_context.newton_shrink_min,
            kernel_context.newton_shrink_max,
        )
        slow_contraction = jnp.logical_and(
            predictor_active,
            jnp.where(
                theta_candidate < kernel_context.theta_diverge_threshold,
                predicted_defect >= jnp.asarray(1.0, dtype=kernel_context.dtype),
                jnp.asarray(True),
            ),
        )
        convergence_metric = jnp.where(theta_valid, faccon * current_newton_norm, current_newton_norm)
        meets_newton_tol = convergence_metric <= kernel_context.predictor_fnewt
        predictor_shrink = jnp.where(
            theta_candidate < kernel_context.theta_diverge_threshold,
            predictor_shrink,
            jnp.asarray(0.5, dtype=kernel_context.dtype),
        )
        shrink_suggest_next = jnp.where(slow_contraction, predictor_shrink, shrink_suggest)
        slow_reject = jnp.logical_and(slow_contraction, jnp.logical_not(meets_newton_tol))
        diverged_next = jnp.logical_or(diverged, jnp.logical_or(slow_reject, jnp.logical_or(residual_blowup, nonfinite_state)))
        if kernel_context.debug_newton_trace:
            jax.debug.print(
                "[radau-solver] iter={iter} delta_norm={delta_norm:.6e} residual_norm={residual_norm:.6e} newton_metric={newton_metric:.6e} fnewt={fnewt:.6e} theta={theta:.6e} slow={slow} blowup={blowup} nonfinite={nonfinite} diverged={diverged}",
                iter=iter_idx + 1,
                delta_norm=current_delta_norm,
                residual_norm=current_residual_norm,
                newton_metric=convergence_metric,
                fnewt=kernel_context.predictor_fnewt,
                theta=theta_next,
                slow=slow_contraction,
                blowup=residual_blowup,
                nonfinite=nonfinite_state,
                diverged=diverged_next,
            )
        return (
            iter_idx + 1,
            z_next,
            current_delta_norm,
            current_residual_norm,
            current_newton_norm,
            convergence_metric,
            theta_ratio_next,
            theta_next,
            diverged_next,
            shrink_suggest_next,
            jnp.logical_or(slow_contraction_any, slow_contraction),
            jnp.logical_or(residual_blowup_any, residual_blowup),
            jnp.logical_or(newton_nonfinite_any, nonfinite_state),
        )

    def cond_fn(newton_state):
        iter_idx, _, delta_norm, residual_norm, _, newton_metric, _, _, diverged, _, _, _, _ = newton_state
        active = jnp.where(
            kernel_context.use_hairer_newton_tol,
            newton_metric > kernel_context.predictor_fnewt,
            jnp.logical_or(residual_norm > kernel_context.tol, delta_norm > kernel_context.tol),
        )
        return jnp.logical_and(jnp.logical_and(iter_idx < kernel_context.maxiter, active), jnp.logical_not(diverged))

    init_newton = (
        jnp.asarray(0, dtype=jnp.int32),
        inputs.z0,
        jnp.asarray(jnp.inf, dtype=kernel_context.dtype),
        jnp.asarray(jnp.inf, dtype=kernel_context.dtype),
        jnp.asarray(jnp.inf, dtype=kernel_context.dtype),
        jnp.asarray(jnp.inf, dtype=kernel_context.dtype),
        kernel_context.zero_scalar,
        kernel_context.zero_scalar,
        jnp.asarray(False),
        jnp.asarray(1.0, dtype=kernel_context.dtype),
        jnp.asarray(False),
        jnp.asarray(False),
        jnp.asarray(False),
    )
    initial_residual = _radau_stage_subsolve_residual(
        kernel_context,
        physics_context,
        inputs,
        inputs.z0,
    )
    finite_initial_residual = jnp.all(jnp.isfinite(initial_residual))
    (
        iter_final,
        z_final,
        delta_norm_final,
        _residual_norm_loop_final,
        _prev_newton_norm_final,
        newton_metric_final,
        _prev_theta_ratio_final,
        theta_final,
        diverged_final,
        shrink_suggest_final,
        slow_contraction_final,
        residual_blowup_final,
        newton_nonfinite_final,
    ) = jax.lax.while_loop(cond_fn, body_fn, init_newton)
    final_residual = _radau_stage_subsolve_residual(
        kernel_context,
        physics_context,
        inputs,
        z_final,
    )
    nonfinite_stage_state = jnp.logical_not(jnp.all(jnp.isfinite(z_final)))
    nonfinite_stage_residual = jnp.logical_not(jnp.all(jnp.isfinite(final_residual)))
    final_residual_norm = _radau_residual_norm(kernel_context, final_residual)
    converged = jnp.logical_and(
        jnp.logical_and(
            jnp.all(jnp.isfinite(z_final)),
            jnp.where(
                kernel_context.use_hairer_newton_tol,
                newton_metric_final <= kernel_context.predictor_fnewt,
                final_residual_norm <= kernel_context.tol,
            ),
        ),
        jnp.logical_not(diverged_final),
    )
    return _RadauStageSubsolveResult(
        iter_final=iter_final,
        z_final=z_final,
        delta_norm_final=delta_norm_final,
        newton_metric_final=newton_metric_final,
        theta_final=theta_final,
        diverged_final=diverged_final,
        shrink_suggest_final=shrink_suggest_final,
        slow_contraction_final=slow_contraction_final,
        residual_blowup_final=residual_blowup_final,
        newton_nonfinite_final=newton_nonfinite_final,
        finite_initial_residual=finite_initial_residual,
        nonfinite_stage_state=nonfinite_stage_state,
        nonfinite_stage_residual=nonfinite_stage_residual,
        final_residual=final_residual,
        final_residual_norm=final_residual_norm,
        converged=converged,
    )


def _radau_single_step_primal(
    kernel_context: _RadauAcceptedStepKernelContext,
    physics_context: _RadauAcceptedStepPhysicsContext,
    carry_in: _RadauAcceptedStepCarry,
    h_value,
):
    flat_y = carry_in.y
    t_value = carry_in.t
    prev_stages = carry_in.prev_stages
    prev_dt = carry_in.prev_dt
    prev_theta_final = carry_in.prev_theta_final
    prev_newton_iter_count = carry_in.prev_newton_iter_count
    jacobian_cache = carry_in.jacobian
    cache_valid = carry_in.cache_valid
    cache_dt = carry_in.cache_dt
    cache_age = carry_in.cache_age
    real_lu_cache = carry_in.real_lu
    real_piv_cache = carry_in.real_piv
    complex_lu_cache = carry_in.complex_lu
    complex_piv_cache = carry_in.complex_piv
    lagged_response, lagged_reference_y, lagged_response_reused = _radau_prepare_lagged_response(
        kernel_context,
        carry_in,
        physics_context.unpack_flat,
        physics_context.project_flat,
        physics_context.build_lagged_response,
    )

    def _rhs_eval(t_eval, y_eval):
        return _radau_eval_rhs(
            t_eval,
            y_eval,
            lagged_response,
            physics_context.flat_rhs,
            physics_context.flat_rhs_with_lagged_response,
        )

    def _rhs_eval_at_current_time(y_eval):
        return _rhs_eval(t_value, y_eval)

    f0 = _rhs_eval(t_value, flat_y)
    z0 = _make_radau_stage_predictor(
        f0,
        prev_stages,
        prev_dt,
        h_value,
        kernel_context.c,
        kernel_context.dtype,
        density_size=kernel_context.density_size,
        pressure_size=kernel_context.pressure_size,
        er_size=kernel_context.er_size,
        prev_theta_final=prev_theta_final,
        prev_newton_iter_count=prev_newton_iter_count,
        predictor_mode=kernel_context.predictor_mode,
    )
    finite_f0 = jnp.all(jnp.isfinite(f0))
    finite_z0 = jnp.all(jnp.isfinite(z0))

    jacobian_dt_scale = jnp.maximum(
        jnp.abs(cache_dt),
        jnp.asarray(1.0e-14, dtype=kernel_context.dtype),
    )
    dt_close = jnp.abs(h_value - cache_dt) <= kernel_context.jacobian_reuse_rtol * jacobian_dt_scale
    reuse_linearization = jnp.logical_and(
        jnp.logical_and(cache_valid, dt_close),
        jnp.logical_not(kernel_context.use_lagged_linear_response),
    )

    def _reuse_linearization(_):
        return jacobian_cache, real_lu_cache, real_piv_cache, complex_lu_cache, complex_piv_cache

    def _recompute_linearization(_):
        jacobian_ref = jax.jacfwd(_rhs_eval_at_current_time)(flat_y)
        h_jacobian = h_value * jacobian_ref
        real_matrix = kernel_context.identity_n - kernel_context.radau_real_eig * h_jacobian
        real_lu, real_piv = jax.scipy.linalg.lu_factor(real_matrix)
        complex_dense_all = jnp.transpose(
            kernel_context.identity_2[None, :, :, None, None] * kernel_context.identity_n[None, None, None, :, :]
            - kernel_context.radau_complex_blocks[:, :, :, None, None] * h_jacobian[None, None, None, :, :],
            (0, 1, 3, 2, 4),
        ).reshape((kernel_context.num_complex_pairs, kernel_context.complex_dim, kernel_context.complex_dim))

        def _factor_pair(i, carry):
            lu_all, piv_all = carry
            lu_i, piv_i = jax.scipy.linalg.lu_factor(complex_dense_all[i])
            lu_all = lu_all.at[i].set(lu_i)
            piv_all = piv_all.at[i].set(piv_i)
            return lu_all, piv_all

        complex_lu, complex_piv = jax.lax.fori_loop(
            0,
            kernel_context.num_complex_pairs,
            _factor_pair,
            (jnp.zeros_like(complex_lu_cache), jnp.zeros_like(complex_piv_cache)),
        )
        return jacobian_ref, real_lu, real_piv, complex_lu, complex_piv

    jacobian_ref, real_lu_out, real_piv_out, complex_lu_out, complex_piv_out = jax.lax.cond(
        reuse_linearization,
        _reuse_linearization,
        _recompute_linearization,
        operand=None,
    )
    jacobian_reused = reuse_linearization

    subsolve_inputs = _radau_build_stage_subsolve_inputs(
        flat_y=flat_y,
        t_value=t_value,
        h_value=h_value,
        z0=z0,
        f0=f0,
        jacobian_ref=jacobian_ref,
        lagged_response=lagged_response,
        real_lu_out=real_lu_out,
        real_piv_out=real_piv_out,
        complex_lu_out=complex_lu_out,
        complex_piv_out=complex_piv_out,
    )
    subsolve_result = _radau_run_stage_subsolve_from_inputs(
        kernel_context,
        physics_context,
        subsolve_inputs,
    )
    stages_final = subsolve_result.z_final.reshape((kernel_context.num_stages, kernel_context.state_dim))
    if kernel_context.debug_newton_trace:
        jax.debug.print(
            "[radau-solver] final iter={iter} converged={converged} diverged={diverged} finite_initial_residual={finite_initial_residual} nonfinite_stage_state={nonfinite_stage_state} nonfinite_stage_residual={nonfinite_stage_residual} residual_norm={residual_norm:.6e} delta_norm={delta_norm:.6e} newton_metric={newton_metric:.6e} fnewt={fnewt:.6e} theta={theta:.6e} slow={slow} blowup={blowup} newton_nonfinite={newton_nonfinite}",
            iter=subsolve_result.iter_final,
            converged=subsolve_result.converged,
            diverged=subsolve_result.diverged_final,
            finite_initial_residual=subsolve_result.finite_initial_residual,
            nonfinite_stage_state=subsolve_result.nonfinite_stage_state,
            nonfinite_stage_residual=subsolve_result.nonfinite_stage_residual,
            residual_norm=subsolve_result.final_residual_norm,
            delta_norm=subsolve_result.delta_norm_final,
            newton_metric=subsolve_result.newton_metric_final,
            fnewt=kernel_context.predictor_fnewt,
            theta=subsolve_result.theta_final,
            slow=subsolve_result.slow_contraction_final,
            blowup=subsolve_result.residual_blowup_final,
            newton_nonfinite=subsolve_result.newton_nonfinite_final,
        )
    flat_next = flat_y + h_value * (kernel_context.b @ stages_final)
    err_vec = h_value * (kernel_context.embedded_f0_weight * f0 + (kernel_context.b_error @ stages_final))
    scale_override = None
    if kernel_context.error_scale_mode == "ntss_transport" and (
        kernel_context.density_size + kernel_context.pressure_size + kernel_context.er_size == kernel_context.state_dim
    ):
        density_end = kernel_context.density_size
        pressure_end = kernel_context.density_size + kernel_context.pressure_size
        density_next = flat_next[:density_end]
        pressure_next = flat_next[density_end:pressure_end]
        er_next = flat_next[pressure_end:pressure_end + kernel_context.er_size]
        er_rms = jnp.sqrt(jnp.mean(er_next * er_next) + jnp.asarray(1.0e-30, dtype=kernel_context.dtype))
        er_floor = jnp.maximum(
            jnp.asarray(0.1, dtype=kernel_context.dtype) * er_rms,
            jnp.asarray(1.0e-3, dtype=kernel_context.dtype),
        )
        density_scale = kernel_context.atol + kernel_context.estimator_rtol_eff * jnp.abs(density_next)
        pressure_scale = kernel_context.atol + kernel_context.estimator_rtol_eff * jnp.abs(pressure_next)
        er_scale = kernel_context.atol + kernel_context.estimator_rtol_eff * jnp.maximum(jnp.abs(er_next), er_floor)
        scale_override = jnp.concatenate([density_scale, pressure_scale, er_scale], axis=0)
    if kernel_context.error_scale_mode in {"ntss_block_floor", "ntss_block_rms"} and (
        kernel_context.density_size + kernel_context.pressure_size + kernel_context.er_size == kernel_context.state_dim
    ):
        density_end = kernel_context.density_size
        pressure_end = kernel_context.density_size + kernel_context.pressure_size
        density_next = flat_next[:density_end]
        pressure_next = flat_next[density_end:pressure_end]
        er_next = flat_next[pressure_end:pressure_end + kernel_context.er_size]
        density_floor = jnp.maximum(
            jnp.asarray(0.05, dtype=kernel_context.dtype)
            * jnp.sqrt(jnp.mean(density_next * density_next) + jnp.asarray(1.0e-30, dtype=kernel_context.dtype)),
            jnp.asarray(1.0e-4, dtype=kernel_context.dtype),
        )
        pressure_floor = jnp.maximum(
            jnp.asarray(0.05, dtype=kernel_context.dtype)
            * jnp.sqrt(jnp.mean(pressure_next * pressure_next) + jnp.asarray(1.0e-30, dtype=kernel_context.dtype)),
            jnp.asarray(1.0e-4, dtype=kernel_context.dtype),
        )
        er_floor = jnp.maximum(
            jnp.asarray(0.1, dtype=kernel_context.dtype)
            * jnp.sqrt(jnp.mean(er_next * er_next) + jnp.asarray(1.0e-30, dtype=kernel_context.dtype)),
            jnp.asarray(1.0e-3, dtype=kernel_context.dtype),
        )
        density_scale = kernel_context.atol + kernel_context.estimator_rtol_eff * jnp.maximum(jnp.abs(density_next), density_floor)
        pressure_scale = kernel_context.atol + kernel_context.estimator_rtol_eff * jnp.maximum(jnp.abs(pressure_next), pressure_floor)
        er_scale = kernel_context.atol + kernel_context.estimator_rtol_eff * jnp.maximum(jnp.abs(er_next), er_floor)
        scale_override = jnp.concatenate([density_scale, pressure_scale, er_scale], axis=0)
    if scale_override is None:
        ref_abs = jnp.abs(flat_y)
        cand_abs = jnp.abs(flat_next)
        max_scale = jnp.maximum(ref_abs, cand_abs)
        mean_scale = jnp.asarray(0.5, dtype=kernel_context.dtype) * (ref_abs + cand_abs)
        scale_base = jnp.where(
            kernel_context.error_scale_mode == "mean",
            mean_scale,
            jnp.where(
                kernel_context.error_scale_mode == "blend",
                jnp.asarray(0.75, dtype=kernel_context.dtype) * max_scale
                + jnp.asarray(0.25, dtype=kernel_context.dtype) * mean_scale,
                jnp.where(
                    kernel_context.error_scale_mode == "ntss",
                    cand_abs,
                    jnp.where(
                        kernel_context.error_scale_mode == "ntss_max",
                        max_scale,
                        jnp.where(
                            kernel_context.error_scale_mode == "ntss_blend",
                            jnp.asarray(0.5, dtype=kernel_context.dtype) * cand_abs
                            + jnp.asarray(0.5, dtype=kernel_context.dtype) * max_scale,
                            max_scale,
                        ),
                    ),
                ),
            ),
        )
        local_scale = kernel_context.atol + kernel_context.estimator_rtol_eff * scale_base
    else:
        local_scale = scale_override
    normalized_err = err_vec / local_scale
    if kernel_context.density_size + kernel_context.pressure_size + kernel_context.er_size == kernel_context.state_dim:
        density_slice = normalized_err[:kernel_context.density_size]
        pressure_slice = normalized_err[
            kernel_context.density_size:kernel_context.density_size + kernel_context.pressure_size
        ]
        er_slice = normalized_err[
            kernel_context.density_size + kernel_context.pressure_size:
            kernel_context.density_size + kernel_context.pressure_size + kernel_context.er_size
        ]
        density_err_norm = jnp.where(
            kernel_context.density_size > 0,
            jnp.sqrt(jnp.mean(density_slice * density_slice) + jnp.asarray(1.0e-30, dtype=kernel_context.dtype)),
            jnp.asarray(0.0, dtype=kernel_context.dtype),
        )
        pressure_err_norm = jnp.where(
            kernel_context.pressure_size > 0,
            jnp.sqrt(jnp.mean(pressure_slice * pressure_slice) + jnp.asarray(1.0e-30, dtype=kernel_context.dtype)),
            jnp.asarray(0.0, dtype=kernel_context.dtype),
        )
        er_err_norm = jnp.where(
            kernel_context.er_size > 0,
            jnp.sqrt(jnp.mean(er_slice * er_slice) + jnp.asarray(1.0e-30, dtype=kernel_context.dtype)),
            jnp.asarray(0.0, dtype=kernel_context.dtype),
        )
    else:
        density_err_norm = jnp.asarray(0.0, dtype=kernel_context.dtype)
        pressure_err_norm = jnp.asarray(0.0, dtype=kernel_context.dtype)
        er_err_norm = jnp.asarray(0.0, dtype=kernel_context.dtype)
    err_norm = _solver_error_norm(
        err_vec,
        flat_y,
        flat_next,
        kernel_context.atol,
        kernel_context.rtol,
        scale_mode=kernel_context.error_scale_mode,
        rtol_eff=kernel_context.estimator_rtol_eff,
        scale_override=scale_override,
    )
    if kernel_context.error_scale_mode == "ntss_block_rms" and (
        kernel_context.density_size + kernel_context.pressure_size + kernel_context.er_size == kernel_context.state_dim
    ):
        active_block_squares = jnp.stack(
            [
                jnp.where(kernel_context.density_size > 0, density_err_norm * density_err_norm, jnp.asarray(0.0, dtype=kernel_context.dtype)),
                jnp.where(kernel_context.pressure_size > 0, pressure_err_norm * pressure_err_norm, jnp.asarray(0.0, dtype=kernel_context.dtype)),
                jnp.where(kernel_context.er_size > 0, er_err_norm * er_err_norm, jnp.asarray(0.0, dtype=kernel_context.dtype)),
            ],
            axis=0,
        )
        active_block_count = (
            jnp.where(kernel_context.density_size > 0, jnp.asarray(1.0, dtype=kernel_context.dtype), jnp.asarray(0.0, dtype=kernel_context.dtype))
            + jnp.where(kernel_context.pressure_size > 0, jnp.asarray(1.0, dtype=kernel_context.dtype), jnp.asarray(0.0, dtype=kernel_context.dtype))
            + jnp.where(kernel_context.er_size > 0, jnp.asarray(1.0, dtype=kernel_context.dtype), jnp.asarray(0.0, dtype=kernel_context.dtype))
        )
        err_norm = jnp.sqrt(
            jnp.sum(active_block_squares) / jnp.maximum(active_block_count, jnp.asarray(1.0, dtype=kernel_context.dtype))
            + jnp.asarray(1.0e-30, dtype=kernel_context.dtype)
        )
    else:
        pressure_err_norm = err_norm
    theta_safe = jnp.clip(subsolve_result.theta_final, kernel_context.theta_clip_min, kernel_context.theta_clip_max)
    fallback_newton_shrink = jnp.clip(
        kernel_context.newton_shrink_num / theta_safe,
        kernel_context.newton_shrink_min,
        kernel_context.newton_shrink_max,
    )
    newton_shrink = jnp.where(
        subsolve_result.converged,
        jnp.asarray(1.0, dtype=kernel_context.dtype),
        jnp.where(subsolve_result.slow_contraction_final, subsolve_result.shrink_suggest_final, fallback_newton_shrink),
    )
    jacobian_out = jacobian_ref
    cache_valid_out = jnp.asarray(True)
    cache_dt_out = h_value
    cache_age_out = jnp.where(reuse_linearization, cache_age + 1, jnp.asarray(0, dtype=jnp.int32))
    return (
        flat_next,
        err_norm,
        subsolve_result.converged,
        subsolve_result.z_final,
        subsolve_result.theta_final,
        subsolve_result.iter_final,
        subsolve_result.final_residual_norm,
        subsolve_result.delta_norm_final,
        subsolve_result.slow_contraction_final,
        subsolve_result.residual_blowup_final,
        subsolve_result.newton_nonfinite_final,
        jacobian_out,
        cache_valid_out,
        cache_dt_out,
        cache_age_out,
        real_lu_out,
        real_piv_out,
        complex_lu_out,
        complex_piv_out,
        newton_shrink,
        subsolve_result.diverged_final,
        subsolve_result.nonfinite_stage_state,
        subsolve_result.nonfinite_stage_residual,
        finite_f0,
        finite_z0,
        subsolve_result.finite_initial_residual,
        density_err_norm,
        pressure_err_norm,
        er_err_norm,
        lagged_response,
        lagged_reference_y,
        lagged_response_reused,
        jacobian_reused,
    )


def _radau_fixed_dt_accepted_rollout(
    kernel_context: _RadauAcceptedStepKernelContext,
    physics_context: _RadauAcceptedStepPhysicsContext,
    carry0: _RadauAcceptedStepCarry,
    dt_sequence,
) -> _RadauAcceptedRolloutResult:
    """AD-facing accepted-rollout helper over an explicit fixed `dt` sequence.

    This helper is intentionally separate from the production adaptive loop.
    It advances the accepted-step carry by replaying a caller-provided `dt`
    sequence and updating the solver carry as if each step were accepted.

    The purpose is to provide a cleaner differentiable map than the raw
    adaptive production trace, without changing production forward behavior.
    """

    def _scan_body(carry, dt_value):
        carry_for_step = dataclasses.replace(carry, dt=dt_value)
        attempt_context = _RadauAcceptedStepAttemptContext(
            t_final=carry.t + dt_value,
            use_transport_lagged_response=jnp.asarray(kernel_context.use_transport_lagged_response),
        )
        step_map_result = _radau_apply_accepted_step_map(
            kernel_context,
            physics_context,
            carry_for_step,
            attempt_context,
        )
        next_carry = dataclasses.replace(
            step_map_result.next_carry,
            prev_error=jnp.maximum(
                step_map_result.err_norm,
                jnp.asarray(1.0e-12, dtype=kernel_context.dtype),
            ),
            recent_reject_count=jnp.asarray(0, dtype=jnp.int32),
            regrowth_cooldown=jnp.asarray(0, dtype=jnp.int32),
            easy_growth_streak=jnp.asarray(0, dtype=jnp.int32),
        )
        scan_out = (
            step_map_result.accepted_y,
            step_map_result.err_norm,
            step_map_result.converged,
            step_map_result.accepted_dt,
        )
        return next_carry, scan_out

    final_carry, scan_outputs = jax.lax.scan(_scan_body, carry0, dt_sequence)
    trial_ys, err_norms, converged_mask, accepted_dts = scan_outputs
    return _RadauAcceptedRolloutResult(
        final_carry=final_carry,
        final_y=final_carry.y,
        trial_ys=trial_ys,
        err_norms=err_norms,
        converged_mask=converged_mask,
        accepted_dts=accepted_dts,
    )


def _radau_replay_realized_accepted_rollout(
    kernel_context: _RadauAcceptedStepKernelContext,
    physics_context: _RadauAcceptedStepPhysicsContext,
    carry0: _RadauAcceptedStepCarry,
    accepted_mask,
    dt_sequence,
) -> _RadauAcceptedRolloutResult:
    """Replay only the realized accepted steps along a fixed forward schedule."""

    dtype = kernel_context.dtype

    def _scan_body(carry, xs):
        accepted, dt_value = xs

        def _do_step(_):
            carry_for_step = dataclasses.replace(carry, dt=dt_value)
            attempt_context = _RadauAcceptedStepAttemptContext(
                t_final=carry.t + dt_value,
                use_transport_lagged_response=jnp.asarray(kernel_context.use_transport_lagged_response),
            )
            step_map_result = _radau_apply_accepted_step_map(
                kernel_context,
                physics_context,
                carry_for_step,
                attempt_context,
            )
            next_carry = dataclasses.replace(
                step_map_result.next_carry,
                prev_error=jnp.maximum(
                    step_map_result.err_norm,
                    jnp.asarray(1.0e-12, dtype=dtype),
                ),
                recent_reject_count=jnp.asarray(0, dtype=jnp.int32),
                regrowth_cooldown=jnp.asarray(0, dtype=jnp.int32),
                easy_growth_streak=jnp.asarray(0, dtype=jnp.int32),
            )
            scan_out = (
                step_map_result.accepted_y,
                step_map_result.err_norm,
                step_map_result.converged,
                dt_value,
            )
            return next_carry, scan_out

        def _skip(_):
            scan_out = (
                carry.y,
                jnp.asarray(jnp.inf, dtype=dtype),
                jnp.asarray(False),
                jnp.asarray(0.0, dtype=dtype),
            )
            return carry, scan_out

        return jax.lax.cond(accepted, _do_step, _skip, operand=None)

    final_carry, scan_outputs = jax.lax.scan(_scan_body, carry0, (accepted_mask, dt_sequence))
    trial_ys, err_norms, converged_mask, accepted_dts = scan_outputs
    return _RadauAcceptedRolloutResult(
        final_carry=final_carry,
        final_y=final_carry.y,
        trial_ys=trial_ys,
        err_norms=err_norms,
        converged_mask=converged_mask,
        accepted_dts=accepted_dts,
    )


def _radau_replay_realized_attempt_rollout(
    execution_context: _RadauSolveExecutionContext,
    carry0: _RadauAcceptedStepCarry,
    active_mask,
    accepted_mask,
    attempted_dts,
    next_dts,
    next_recent_reject_count,
    next_regrowth_cooldown,
    next_easy_growth_streak,
    next_lagged_response_valid,
) -> _RadauAcceptedRolloutResult:
    """Replay the full realized attempt history using forward trace metadata.

    This keeps the discrete adaptive path fixed to the primal rollout while
    still differentiating the accepted-step physical update on accepted
    attempts.
    """

    dtype = execution_context.dtype

    def _scan_body(carry, xs):
        (
            active,
            accepted,
            dt_value,
            next_dt_value,
            recent_reject_count_value,
            regrowth_cooldown_value,
            easy_growth_streak_value,
            lagged_response_valid_value,
        ) = xs

        def _do_attempt(_):
            def _accepted_attempt(__):
                carry_for_step = dataclasses.replace(carry, dt=dt_value)
                attempt_result = _execute_radau_accepted_step_attempt_autodiff(
                    execution_context.kernel_context,
                    execution_context.physics_context,
                    _radau_carry_with_forward_only_jvp_fields(carry_for_step),
                    execution_context.attempt_context,
                )
                accepted_y = _project_flat_state_if_needed(
                    attempt_result.trial_y,
                    execution_context.physics_context.project_flat,
                )
                next_carry = dataclasses.replace(
                    attempt_result.carry_after_attempt,
                    t=carry.t + dt_value,
                    y=accepted_y,
                    dt=next_dt_value,
                    prev_error=jnp.maximum(
                        attempt_result.err_norm,
                        jnp.asarray(1.0e-12, dtype=dtype),
                    ),
                    prev_stages=attempt_result.stage_history,
                    prev_dt=dt_value,
                    recent_reject_count=recent_reject_count_value,
                    regrowth_cooldown=regrowth_cooldown_value,
                    easy_growth_streak=easy_growth_streak_value,
                    lagged_response_valid=lagged_response_valid_value,
                    jacobian=attempt_result.jacobian_out,
                    cache_valid=attempt_result.cache_valid_out,
                    cache_dt=attempt_result.cache_dt_out,
                    cache_age=attempt_result.cache_age_out,
                    real_lu=attempt_result.real_lu_out,
                    real_piv=attempt_result.real_piv_out,
                    complex_lu=attempt_result.complex_lu_out,
                    complex_piv=attempt_result.complex_piv_out,
                    prev_theta_final=attempt_result.theta_final,
                    prev_newton_iter_count=attempt_result.newton_iter_count,
                )
                scan_out = (
                    next_carry.y,
                    attempt_result.err_norm,
                    attempt_result.converged,
                    dt_value,
                )
                return next_carry, scan_out

            def _rejected_attempt(__):
                carry_for_step = dataclasses.replace(jax.lax.stop_gradient(carry), dt=dt_value)
                attempt_result = _execute_radau_accepted_step_attempt_autodiff(
                    execution_context.kernel_context,
                    execution_context.physics_context,
                    _radau_carry_with_forward_only_jvp_fields(carry_for_step),
                    execution_context.attempt_context,
                )
                next_carry = dataclasses.replace(
                    carry,
                    dt=next_dt_value,
                    recent_reject_count=recent_reject_count_value,
                    regrowth_cooldown=regrowth_cooldown_value,
                    easy_growth_streak=easy_growth_streak_value,
                    lagged_response_cache=jax.lax.stop_gradient(attempt_result.carry_after_attempt.lagged_response_cache),
                    lagged_response_valid=lagged_response_valid_value,
                    lagged_reference_y=jax.lax.stop_gradient(attempt_result.carry_after_attempt.lagged_reference_y),
                    jacobian=jax.lax.stop_gradient(attempt_result.jacobian_out),
                    cache_valid=jax.lax.stop_gradient(attempt_result.cache_valid_out),
                    cache_dt=jax.lax.stop_gradient(attempt_result.cache_dt_out),
                    cache_age=jax.lax.stop_gradient(attempt_result.cache_age_out),
                    real_lu=jax.lax.stop_gradient(attempt_result.real_lu_out),
                    real_piv=jax.lax.stop_gradient(attempt_result.real_piv_out),
                    complex_lu=jax.lax.stop_gradient(attempt_result.complex_lu_out),
                    complex_piv=jax.lax.stop_gradient(attempt_result.complex_piv_out),
                    prev_theta_final=jax.lax.stop_gradient(attempt_result.theta_final),
                    prev_newton_iter_count=jax.lax.stop_gradient(attempt_result.newton_iter_count),
                )
                scan_out = (
                    next_carry.y,
                    jax.lax.stop_gradient(attempt_result.err_norm),
                    jax.lax.stop_gradient(attempt_result.converged),
                    jnp.asarray(0.0, dtype=dtype),
                )
                return next_carry, scan_out

            return jax.lax.cond(accepted, _accepted_attempt, _rejected_attempt, operand=None)

        def _skip(_):
            scan_out = (
                carry.y,
                jnp.asarray(jnp.inf, dtype=dtype),
                jnp.asarray(False),
                jnp.asarray(0.0, dtype=dtype),
            )
            return carry, scan_out

        return jax.lax.cond(active, _do_attempt, _skip, operand=None)

    final_carry, scan_outputs = jax.lax.scan(
        _scan_body,
        carry0,
        (
            active_mask,
            accepted_mask,
            attempted_dts,
            next_dts,
            next_recent_reject_count,
            next_regrowth_cooldown,
            next_easy_growth_streak,
            next_lagged_response_valid,
        ),
    )
    trial_ys, err_norms, converged_mask, accepted_dts = scan_outputs
    return _RadauAcceptedRolloutResult(
        final_carry=final_carry,
        final_y=final_carry.y,
        trial_ys=trial_ys,
        err_norms=err_norms,
        converged_mask=converged_mask,
        accepted_dts=accepted_dts,
    )


def _radau_dt_sequence_from_time_list(
    time_list,
    *,
    t0,
    dtype,
) -> jax.Array:
    """Convert an absolute time list into accepted-step `dt` increments."""

    times_np = np.asarray(time_list, dtype=float).reshape(-1)
    if times_np.size == 0:
        return jnp.asarray([], dtype=dtype)

    t0_value = float(jnp.asarray(t0))
    if np.isclose(times_np[0], t0_value, rtol=0.0, atol=1.0e-14):
        times_np = times_np[1:]
    if times_np.size == 0:
        return jnp.asarray([], dtype=dtype)

    full_times = np.concatenate(([t0_value], times_np), axis=0)
    dt_sequence_np = np.diff(full_times)
    if np.any(dt_sequence_np < -1.0e-14):
        raise ValueError("time_list must be nondecreasing and not go backward in time.")
    return jnp.asarray(dt_sequence_np, dtype=dtype)


def _radau_run_prepared_on_time_list(
    prepared_rollout: _PreparedRadauAcceptedRollout,
    time_list,
) -> dict[str, Any]:
    """Replay the accepted-step Radau map on a caller-provided absolute time list."""

    dt_sequence = _radau_dt_sequence_from_time_list(
        time_list,
        t0=prepared_rollout.kernel_context.t0,
        dtype=prepared_rollout.kernel_context.dtype,
    )
    rollout = _radau_fixed_dt_accepted_rollout(
        prepared_rollout.kernel_context,
        prepared_rollout.physics_context,
        prepared_rollout.initial_carry,
        dt_sequence,
    )
    saved_states = jax.vmap(prepared_rollout.physics_context.unpack_flat)(rollout.trial_ys)
    final_state = prepared_rollout.physics_context.unpack_flat(rollout.final_carry.y)
    return {
        "time_list": jnp.asarray(time_list, dtype=prepared_rollout.kernel_context.dtype),
        "dt_sequence": dt_sequence,
        "saved_states": saved_states,
        "final_state": final_state,
        "final_carry": rollout.final_carry,
        "rollout": rollout,
    }


def _radau_run_prepared_on_realized_trace(
    prepared_rollout: _PreparedRadauAcceptedRollout,
    execution_context: _RadauSolveExecutionContext,
    trace: _RadauAdaptiveRolloutTrace,
    *,
    replay_mode: str = "attempt",
) -> dict[str, Any]:
    """Replay a prepared Radau rollout on a frozen forward trace from another solve."""

    replay_mode_normalized = str(replay_mode).strip().lower()
    if replay_mode_normalized == "attempt":
        rollout = _radau_replay_realized_attempt_rollout(
            execution_context,
            prepared_rollout.initial_carry,
            jax.lax.stop_gradient(trace.active_mask),
            jax.lax.stop_gradient(trace.accepted_mask),
            jax.lax.stop_gradient(trace.attempted_dts),
            jax.lax.stop_gradient(trace.next_dts),
            jax.lax.stop_gradient(trace.next_recent_reject_count),
            jax.lax.stop_gradient(trace.next_regrowth_cooldown),
            jax.lax.stop_gradient(trace.next_easy_growth_streak),
            jax.lax.stop_gradient(trace.next_lagged_response_valid),
        )
    elif replay_mode_normalized == "accepted":
        rollout = _radau_replay_realized_accepted_rollout(
            prepared_rollout.kernel_context,
            prepared_rollout.physics_context,
            prepared_rollout.initial_carry,
            jax.lax.stop_gradient(trace.accepted_mask),
            jax.lax.stop_gradient(trace.attempted_dts),
        )
    else:
        raise ValueError(
            "replay_mode must be one of {'attempt', 'accepted'} "
            f"but got {replay_mode!r}."
        )

    final_state = prepared_rollout.physics_context.unpack_flat(rollout.final_carry.y)
    return {
        "final_state": final_state,
        "final_carry": rollout.final_carry,
        "rollout": rollout,
        "replay_mode": replay_mode_normalized,
    }


def _radau_controller_composed_rollout(
    execution_context: _RadauSolveExecutionContext,
    carry0: _RadauAcceptedStepCarry,
    *,
    step_count: int,
    dt_scale: float | Any = 1.0,
) -> _RadauControllerRolloutResult:
    """AD-facing rollout with real controller updates but without the outer saved loop."""

    dtype = execution_context.dtype
    scaled_dt = jnp.asarray(dt_scale, dtype=dtype) * carry0.dt
    step_state0 = _radau_step_state_from_carry(
        dataclasses.replace(carry0, dt=scaled_dt),
        status=jnp.asarray([0, 0, 0], dtype=jnp.int32),
    )

    def _scan_body(step_state, _):
        next_step_state, step_info = _radau_step_fn(execution_context, step_state, None)
        scan_out = (
            next_step_state.y,
            step_info.err_norm,
            step_info.accepted,
            step_info.dt,
            step_info.next_dt,
        )
        return next_step_state, scan_out

    final_step_state, scan_outputs = jax.lax.scan(
        _scan_body,
        step_state0,
        xs=jnp.arange(int(step_count), dtype=jnp.int32),
    )
    step_ys, err_norms, accepted_mask, attempted_dts, next_dts = scan_outputs
    final_carry = _radau_carry_from_step_state(final_step_state)
    return _RadauControllerRolloutResult(
        final_step_state=final_step_state,
        final_carry=final_carry,
        step_ys=step_ys,
        err_norms=err_norms,
        accepted_mask=accepted_mask,
        attempted_dts=attempted_dts,
        next_dts=next_dts,
    )


def _radau_controller_forward_only_rollout(
    execution_context: _RadauSolveExecutionContext,
    carry0: _RadauAcceptedStepCarry,
    *,
    step_count: int,
    dt_scale: float | Any = 1.0,
) -> _RadauControllerRolloutResult:
    """Rollout with differentiated Radau steps but forward-only controller evolution."""

    dtype = execution_context.dtype
    scaled_dt = jnp.asarray(dt_scale, dtype=dtype) * carry0.dt
    step_state0 = _radau_step_state_with_forward_only_controller_fields(
        _radau_step_state_from_carry(
            dataclasses.replace(carry0, dt=scaled_dt),
            status=jnp.asarray([0, 0, 0], dtype=jnp.int32),
        )
    )

    def _scan_body(step_state, _):
        next_step_state_raw, step_info = _radau_step_fn(execution_context, step_state, None)
        next_step_state = _radau_step_state_with_forward_only_controller_fields(next_step_state_raw)
        scan_out = (
            next_step_state_raw.y,
            step_info.err_norm,
            step_info.accepted,
            step_info.dt,
            step_info.next_dt,
        )
        return next_step_state, scan_out

    final_step_state, scan_outputs = jax.lax.scan(
        _scan_body,
        step_state0,
        xs=jnp.arange(int(step_count), dtype=jnp.int32),
    )
    step_ys, err_norms, accepted_mask, attempted_dts, next_dts = scan_outputs
    final_carry = _radau_carry_from_step_state(final_step_state)
    return _RadauControllerRolloutResult(
        final_step_state=final_step_state,
        final_carry=final_carry,
        step_ys=step_ys,
        err_norms=err_norms,
        accepted_mask=accepted_mask,
        attempted_dts=attempted_dts,
        next_dts=next_dts,
    )


def _radau_adaptive_final_state_rollout(
    execution_context: _RadauSolveExecutionContext,
    carry0: _RadauAcceptedStepCarry,
    *,
    max_total_steps: int,
    stop_after_accepted_steps: int | None = None,
) -> _RadauAdaptiveRolloutResult:
    """Run the real adaptive Radau rollout and record a minimal per-attempt trace.

    This is the intended solve-level AD boundary for future custom rollout JVP
    work. Forward behavior follows the production adaptive controller/acceptance
    logic; the extra output is only the compact trace needed to define that
    boundary explicitly.
    """

    dtype = execution_context.dtype
    step_state0 = _radau_step_state_from_carry(
        carry0,
        status=jnp.asarray([0, 0, 0], dtype=jnp.int32),
    )

    def _inactive_payload(step_state: _RadauStepState):
        carry = _radau_carry_from_step_state(step_state)
        return _RadauAcceptedStepBackwardPayloadCandidate(
            t_start=carry.t,
            y_start=carry.y,
            dt=jnp.asarray(0.0, dtype=dtype),
            prev_stages=carry.prev_stages,
            prev_dt=jnp.asarray(0.0, dtype=dtype),
            prev_theta_final=carry.prev_theta_final,
            prev_newton_iter_count=carry.prev_newton_iter_count,
            lagged_response_cache=carry.lagged_response_cache,
            lagged_response_valid=carry.lagged_response_valid,
            lagged_reference_y=carry.lagged_reference_y,
            y_end=carry.y,
        )

    def _inactive_step_info(step_state: _RadauStepState):
        failed = step_state.status[0] != 0
        fail_code = step_state.status[1]
        return _RadauStepInfo(
            y=step_state.y,
            t=step_state.t,
            dt=jnp.asarray(0.0, dtype=dtype),
            next_dt=step_state.dt,
            growth=jnp.asarray(1.0, dtype=dtype),
            lagged_reused=jnp.asarray(False),
            jacobian_reused=jnp.asarray(False),
            accepted=jnp.asarray(False),
            failed=failed,
            fail_code=fail_code,
            converged=jnp.asarray(False),
            err_norm=jnp.asarray(jnp.inf, dtype=dtype),
            diverged=jnp.asarray(False),
            nonfinite_stage_state=jnp.asarray(False),
            nonfinite_stage_residual=jnp.asarray(False),
            finite_f0=jnp.asarray(True),
            finite_z0=jnp.asarray(True),
            finite_initial_residual=jnp.asarray(True),
            newton_iter_count=jnp.asarray(0, dtype=jnp.int32),
            final_residual_norm=jnp.asarray(jnp.inf, dtype=dtype),
            final_delta_norm=jnp.asarray(jnp.inf, dtype=dtype),
            theta_final=jnp.asarray(0.0, dtype=dtype),
            slow_contraction=jnp.asarray(False),
            residual_blowup=jnp.asarray(False),
            newton_nonfinite=jnp.asarray(False),
        )

    xs = jnp.arange(int(max_total_steps), dtype=jnp.int32)

    def _scan_body_with_idx(step_state, step_idx):
        active = jnp.logical_and(
            _custom_loop_active(
                step_state,
                execution_context.attempt_context.t_final,
                step_idx,
                max_total_steps,
            ),
            jnp.logical_not(_accepted_step_limit_reached(step_state, stop_after_accepted_steps)),
        )

        def _run(_):
            return _radau_attempt_step_with_payload(execution_context, step_state)

        def _skip(_):
            return step_state, _inactive_step_info(step_state), _inactive_payload(step_state)

        next_step_state, step_info, payload = jax.lax.cond(active, _run, _skip, operand=None)
        scan_out = (
            active,
            jnp.asarray(step_info.accepted),
            payload,
            jnp.asarray(step_info.err_norm, dtype=dtype),
            jnp.asarray(0.0 if getattr(step_info, "theta_final", None) is None else getattr(step_info, "theta_final"), dtype=dtype),
            jnp.asarray(0 if getattr(step_info, "newton_iter_count", None) is None else getattr(step_info, "newton_iter_count"), dtype=jnp.int32),
            jnp.asarray(next_step_state.cache_valid),
            jnp.asarray(step_info.dt, dtype=dtype),
            jnp.asarray(step_info.next_dt, dtype=dtype),
            jnp.asarray(step_info.t, dtype=dtype),
            next_step_state.recent_reject_count,
            next_step_state.regrowth_cooldown,
            next_step_state.easy_growth_streak,
            next_step_state.lagged_response_valid,
        )
        return next_step_state, scan_out

    final_step_state, scan_outputs = jax.lax.scan(_scan_body_with_idx, step_state0, xs)
    (
        active_mask,
        accepted_mask,
        payloads,
        err_norms,
        theta_finals,
        newton_iter_counts,
        cache_valid_next,
        attempted_dts,
        next_dts,
        step_ts,
        next_recent_reject_count,
        next_regrowth_cooldown,
        next_easy_growth_streak,
        next_lagged_response_valid,
    ) = scan_outputs
    final_carry = _radau_carry_from_step_state(final_step_state)
    trace = _RadauAdaptiveRolloutTrace(
        accepted_mask=accepted_mask,
        active_mask=active_mask,
        t_start=payloads.t_start,
        y_start=payloads.y_start,
        dt=payloads.dt,
        prev_stages=payloads.prev_stages,
        prev_dt=payloads.prev_dt,
        prev_theta_final=payloads.prev_theta_final,
        prev_newton_iter_count=payloads.prev_newton_iter_count,
        lagged_response_cache=payloads.lagged_response_cache,
        lagged_response_valid=payloads.lagged_response_valid,
        lagged_reference_y=payloads.lagged_reference_y,
        y_end=payloads.y_end,
        err_norms=err_norms,
        theta_finals=theta_finals,
        newton_iter_counts=newton_iter_counts,
        cache_valid_next=cache_valid_next,
        attempted_dts=attempted_dts,
        next_dts=next_dts,
        step_ts=step_ts,
        next_recent_reject_count=next_recent_reject_count,
        next_regrowth_cooldown=next_regrowth_cooldown,
        next_easy_growth_streak=next_easy_growth_streak,
        next_lagged_response_valid=next_lagged_response_valid,
    )
    completed = jnp.logical_or(
        final_step_state.t >= (execution_context.attempt_context.t_final - 1.0e-15),
        _accepted_step_limit_reached(final_step_state, stop_after_accepted_steps),
    )
    failed = final_step_state.status[0] != 0
    fail_code = final_step_state.status[1]
    return _RadauAdaptiveRolloutResult(
        final_step_state=final_step_state,
        final_carry=final_carry,
        trace=trace,
        attempt_count=jnp.sum(active_mask.astype(jnp.int32)),
        accepted_count=jnp.sum(accepted_mask.astype(jnp.int32)),
        completed=completed,
        failed=failed,
        fail_code=fail_code,
    )


@partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2))
def _radau_adaptive_final_y_realized_schedule(
    execution_context: _RadauSolveExecutionContext,
    max_total_steps: int,
    stop_after_accepted_steps: int | None,
    carry0: _RadauAcceptedStepCarry,
):
    """Final adaptive state with a solve-level JVP over the realized accepted schedule."""

    rollout = _radau_adaptive_final_state_rollout(
        execution_context,
        carry0,
        max_total_steps=max_total_steps,
        stop_after_accepted_steps=stop_after_accepted_steps,
    )
    return rollout.final_carry.y


@_radau_adaptive_final_y_realized_schedule.defjvp
def _radau_adaptive_final_y_realized_schedule_jvp(
    execution_context: _RadauSolveExecutionContext,
    max_total_steps: int,
    stop_after_accepted_steps: int | None,
    primals,
    tangents,
):
    (carry0,) = primals
    (carry0_dot,) = tangents
    rollout = _radau_adaptive_final_state_rollout(
        execution_context,
        carry0,
        max_total_steps=max_total_steps,
        stop_after_accepted_steps=stop_after_accepted_steps,
    )
    active_mask = jax.lax.stop_gradient(rollout.trace.active_mask)
    accepted_mask = jax.lax.stop_gradient(rollout.trace.accepted_mask)
    attempted_dts = jax.lax.stop_gradient(rollout.trace.attempted_dts)
    next_dts = jax.lax.stop_gradient(rollout.trace.next_dts)
    next_recent_reject_count = jax.lax.stop_gradient(rollout.trace.next_recent_reject_count)
    next_regrowth_cooldown = jax.lax.stop_gradient(rollout.trace.next_regrowth_cooldown)
    next_easy_growth_streak = jax.lax.stop_gradient(rollout.trace.next_easy_growth_streak)
    next_lagged_response_valid = jax.lax.stop_gradient(rollout.trace.next_lagged_response_valid)

    def _replay(carry_value):
        replay = _radau_replay_realized_attempt_rollout(
            execution_context,
            carry_value,
            active_mask,
            accepted_mask,
            attempted_dts,
            next_dts,
            next_recent_reject_count,
            next_regrowth_cooldown,
            next_easy_growth_streak,
            next_lagged_response_valid,
        )
        return replay.final_carry.y

    primal_out, tangent_out = jax.jvp(_replay, (carry0,), (carry0_dot,))
    return primal_out, tangent_out


def _radau_debug_realized_attempt_replay(
    execution_context: _RadauSolveExecutionContext,
    carry0: _RadauAcceptedStepCarry,
    carry0_dot: _RadauAcceptedStepCarry,
    trace: _RadauAdaptiveRolloutTrace,
) -> _RadauReplayNanDiagnostic:
    """Locate the first nonfinite tangent in the realized-attempt replay."""

    active_mask = trace.active_mask
    accepted_mask = trace.accepted_mask
    attempted_dts = trace.attempted_dts
    next_dts = trace.next_dts
    next_recent_reject_count = trace.next_recent_reject_count
    next_regrowth_cooldown = trace.next_regrowth_cooldown
    next_easy_growth_streak = trace.next_easy_growth_streak
    next_lagged_response_valid = trace.next_lagged_response_valid

    def _tree_all_finite(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        finite = jnp.asarray(True)
        for leaf in leaves:
            leaf_arr = jnp.asarray(leaf)
            if jnp.issubdtype(leaf_arr.dtype, jnp.inexact):
                finite = jnp.logical_and(finite, jnp.all(jnp.isfinite(leaf_arr)))
        return finite

    def _max_abs_scalar(value):
        zero = jnp.asarray(0.0, dtype=execution_context.dtype)
        if value is None:
            return zero
        leaves = jax.tree_util.tree_leaves(value, is_leaf=lambda x: x is None)
        max_abs = zero
        for leaf in leaves:
            if leaf is None:
                continue
            leaf_arr = jnp.asarray(leaf)
            if not jnp.issubdtype(leaf_arr.dtype, jnp.inexact):
                continue
            max_abs = jnp.maximum(max_abs, jnp.max(jnp.abs(leaf_arr)).astype(execution_context.dtype))
        return max_abs

    def _state_component_max_abs(state_value, attr_name: str):
        if state_value is None:
            return jnp.asarray(0.0, dtype=execution_context.dtype)
        if hasattr(state_value, attr_name):
            return _max_abs_scalar(getattr(state_value, attr_name))
        return jnp.asarray(0.0, dtype=execution_context.dtype)

    def _accepted_attempt(
        carry_value,
        *,
        dt_value,
        next_dt_value,
        recent_reject_count_value,
        regrowth_cooldown_value,
        easy_growth_streak_value,
        lagged_response_valid_value,
    ):
        carry_for_step = dataclasses.replace(carry_value, dt=dt_value)
        attempt_result = _execute_radau_accepted_step_attempt_autodiff(
            execution_context.kernel_context,
            execution_context.physics_context,
            _radau_carry_with_forward_only_jvp_fields(carry_for_step),
            execution_context.attempt_context,
        )
        accepted_y = _project_flat_state_if_needed(
            attempt_result.trial_y,
            execution_context.physics_context.project_flat,
        )
        return dataclasses.replace(
            attempt_result.carry_after_attempt,
            t=carry_value.t + dt_value,
            y=accepted_y,
            dt=next_dt_value,
            prev_error=jnp.maximum(
                attempt_result.err_norm,
                jnp.asarray(1.0e-12, dtype=execution_context.dtype),
            ),
            prev_stages=attempt_result.stage_history,
            prev_dt=dt_value,
            recent_reject_count=recent_reject_count_value,
            regrowth_cooldown=regrowth_cooldown_value,
            easy_growth_streak=easy_growth_streak_value,
            lagged_response_valid=lagged_response_valid_value,
            jacobian=attempt_result.jacobian_out,
            cache_valid=attempt_result.cache_valid_out,
            cache_dt=attempt_result.cache_dt_out,
            cache_age=attempt_result.cache_age_out,
            real_lu=attempt_result.real_lu_out,
            real_piv=attempt_result.real_piv_out,
            complex_lu=attempt_result.complex_lu_out,
            complex_piv=attempt_result.complex_piv_out,
            prev_theta_final=attempt_result.theta_final,
            prev_newton_iter_count=attempt_result.newton_iter_count,
        )

    def _rejected_attempt(
        carry_value,
        *,
        dt_value,
        next_dt_value,
        recent_reject_count_value,
        regrowth_cooldown_value,
        easy_growth_streak_value,
        lagged_response_valid_value,
    ):
        carry_for_step = dataclasses.replace(jax.lax.stop_gradient(carry_value), dt=dt_value)
        attempt_result = _execute_radau_accepted_step_attempt_autodiff(
            execution_context.kernel_context,
            execution_context.physics_context,
            _radau_carry_with_forward_only_jvp_fields(carry_for_step),
            execution_context.attempt_context,
        )
        return dataclasses.replace(
            carry_value,
            dt=next_dt_value,
            recent_reject_count=recent_reject_count_value,
            regrowth_cooldown=regrowth_cooldown_value,
            easy_growth_streak=easy_growth_streak_value,
            lagged_response_cache=jax.lax.stop_gradient(attempt_result.carry_after_attempt.lagged_response_cache),
            lagged_response_valid=lagged_response_valid_value,
            lagged_reference_y=jax.lax.stop_gradient(attempt_result.carry_after_attempt.lagged_reference_y),
            jacobian=jax.lax.stop_gradient(attempt_result.jacobian_out),
            cache_valid=jax.lax.stop_gradient(attempt_result.cache_valid_out),
            cache_dt=jax.lax.stop_gradient(attempt_result.cache_dt_out),
            cache_age=jax.lax.stop_gradient(attempt_result.cache_age_out),
            real_lu=jax.lax.stop_gradient(attempt_result.real_lu_out),
            real_piv=jax.lax.stop_gradient(attempt_result.real_piv_out),
            complex_lu=jax.lax.stop_gradient(attempt_result.complex_lu_out),
            complex_piv=jax.lax.stop_gradient(attempt_result.complex_piv_out),
            prev_theta_final=jax.lax.stop_gradient(attempt_result.theta_final),
            prev_newton_iter_count=jax.lax.stop_gradient(attempt_result.newton_iter_count),
        )

    def _scan_body(scan_state, inputs):
        carry, carry_dot, still_finite = scan_state
        (
            active,
            accepted,
            dt_value,
            next_dt_value,
            recent_reject_count_value,
            regrowth_cooldown_value,
            easy_growth_streak_value,
            lagged_response_valid_value,
        ) = inputs

        should_run = jnp.logical_and(active, still_finite)

        def _run_step(step_operand):
            carry_in, carry_dot_in = step_operand

            def _run_accepted(_):
                return jax.jvp(
                    lambda c: _accepted_attempt(
                        c,
                        dt_value=dt_value,
                        next_dt_value=next_dt_value,
                        recent_reject_count_value=recent_reject_count_value,
                        regrowth_cooldown_value=regrowth_cooldown_value,
                        easy_growth_streak_value=easy_growth_streak_value,
                        lagged_response_valid_value=lagged_response_valid_value,
                    ),
                    (carry_in,),
                    (carry_dot_in,),
                )

            def _run_rejected(_):
                return jax.jvp(
                    lambda c: _rejected_attempt(
                        c,
                        dt_value=dt_value,
                        next_dt_value=next_dt_value,
                        recent_reject_count_value=recent_reject_count_value,
                        regrowth_cooldown_value=regrowth_cooldown_value,
                        easy_growth_streak_value=easy_growth_streak_value,
                        lagged_response_valid_value=lagged_response_valid_value,
                    ),
                    (carry_in,),
                    (carry_dot_in,),
                )

            return jax.lax.cond(accepted, _run_accepted, _run_rejected, operand=None)

        def _skip_step(step_operand):
            return step_operand

        carry, carry_dot = jax.lax.cond(
            should_run,
            _run_step,
            _skip_step,
            operand=(carry, carry_dot),
        )
        is_finite = _tree_all_finite(carry_dot)
        dt_dot_abs = _max_abs_scalar(carry_dot.dt)
        prev_error_dot_abs = _max_abs_scalar(carry_dot.prev_error)
        density_dot_max_abs = _state_component_max_abs(carry_dot.y, "density")
        pressure_dot_max_abs = _state_component_max_abs(carry_dot.y, "pressure")
        er_dot_max_abs = _state_component_max_abs(carry_dot.y, "Er")
        y_dot_max_abs = _max_abs_scalar(carry_dot.y)
        prev_stages_dot_max_abs = _max_abs_scalar(carry_dot.prev_stages)
        lagged_response_cache_dot_max_abs = _max_abs_scalar(carry_dot.lagged_response_cache)
        jacobian_dot_max_abs = _max_abs_scalar(carry_dot.jacobian)
        real_lu_dot_max_abs = _max_abs_scalar(carry_dot.real_lu)
        complex_lu_dot_max_abs = _max_abs_scalar(carry_dot.complex_lu)
        return (
            carry,
            carry_dot,
            jnp.logical_and(still_finite, is_finite),
        ), (
            is_finite,
            dt_dot_abs,
            prev_error_dot_abs,
            density_dot_max_abs,
            pressure_dot_max_abs,
            er_dot_max_abs,
            y_dot_max_abs,
            prev_stages_dot_max_abs,
            lagged_response_cache_dot_max_abs,
            jacobian_dot_max_abs,
            real_lu_dot_max_abs,
            complex_lu_dot_max_abs,
        )

    scan_inputs = (
        active_mask,
        accepted_mask,
        attempted_dts,
        next_dts,
        next_recent_reject_count,
        next_regrowth_cooldown,
        next_easy_growth_streak,
        next_lagged_response_valid,
    )
    (_, _, _), scan_outputs = jax.lax.scan(
        _scan_body,
        (carry0, carry0_dot, jnp.asarray(True)),
        scan_inputs,
    )
    (
        tangent_finite_mask,
        dt_dot_abs,
        prev_error_dot_abs,
        density_dot_max_abs,
        pressure_dot_max_abs,
        er_dot_max_abs,
        y_dot_max_abs,
        prev_stages_dot_max_abs,
        lagged_response_cache_dot_max_abs,
        jacobian_dot_max_abs,
        real_lu_dot_max_abs,
        complex_lu_dot_max_abs,
    ) = scan_outputs

    any_bad = jnp.any(jnp.logical_not(tangent_finite_mask))
    first_bad_index = jnp.where(
        any_bad,
        jnp.argmax(jnp.logical_not(tangent_finite_mask)).astype(jnp.int32),
        jnp.asarray(-1, dtype=jnp.int32),
    )
    safe_index = jnp.maximum(first_bad_index, 0)
    first_bad_was_accepted = jnp.where(any_bad, accepted_mask[safe_index], jnp.asarray(False))
    first_bad_dt = jnp.where(
        any_bad,
        attempted_dts[safe_index],
        jnp.asarray(jnp.nan, dtype=attempted_dts.dtype),
    )
    diagnostic = _RadauReplayNanDiagnostic(
        tangent_finite_mask=tangent_finite_mask,
        first_bad_index=first_bad_index,
        first_bad_was_accepted=first_bad_was_accepted,
        first_bad_dt=first_bad_dt,
        final_tangent_finite=tangent_finite_mask[-1] if tangent_finite_mask.shape[0] else jnp.asarray(True),
        dt_dot_abs=dt_dot_abs,
        prev_error_dot_abs=prev_error_dot_abs,
        density_dot_max_abs=density_dot_max_abs,
        pressure_dot_max_abs=pressure_dot_max_abs,
        er_dot_max_abs=er_dot_max_abs,
        y_dot_max_abs=y_dot_max_abs,
        prev_stages_dot_max_abs=prev_stages_dot_max_abs,
        lagged_response_cache_dot_max_abs=lagged_response_cache_dot_max_abs,
        jacobian_dot_max_abs=jacobian_dot_max_abs,
        real_lu_dot_max_abs=real_lu_dot_max_abs,
        complex_lu_dot_max_abs=complex_lu_dot_max_abs,
    )
    diagnostic = jax.device_get(diagnostic)
    return _RadauReplayNanDiagnostic(
        tangent_finite_mask=np.asarray(diagnostic.tangent_finite_mask, dtype=bool).tolist(),
        first_bad_index=int(diagnostic.first_bad_index),
        first_bad_was_accepted=bool(diagnostic.first_bad_was_accepted),
        first_bad_dt=float(diagnostic.first_bad_dt),
        final_tangent_finite=bool(diagnostic.final_tangent_finite),
        dt_dot_abs=np.asarray(diagnostic.dt_dot_abs, dtype=float).tolist(),
        prev_error_dot_abs=np.asarray(diagnostic.prev_error_dot_abs, dtype=float).tolist(),
        density_dot_max_abs=np.asarray(diagnostic.density_dot_max_abs, dtype=float).tolist(),
        pressure_dot_max_abs=np.asarray(diagnostic.pressure_dot_max_abs, dtype=float).tolist(),
        er_dot_max_abs=np.asarray(diagnostic.er_dot_max_abs, dtype=float).tolist(),
        y_dot_max_abs=np.asarray(diagnostic.y_dot_max_abs, dtype=float).tolist(),
        prev_stages_dot_max_abs=np.asarray(diagnostic.prev_stages_dot_max_abs, dtype=float).tolist(),
        lagged_response_cache_dot_max_abs=np.asarray(diagnostic.lagged_response_cache_dot_max_abs, dtype=float).tolist(),
        jacobian_dot_max_abs=np.asarray(diagnostic.jacobian_dot_max_abs, dtype=float).tolist(),
        real_lu_dot_max_abs=np.asarray(diagnostic.real_lu_dot_max_abs, dtype=float).tolist(),
        complex_lu_dot_max_abs=np.asarray(diagnostic.complex_lu_dot_max_abs, dtype=float).tolist(),
    )


def _radau_debug_compare_zero_tangent_one_step(
    execution_context: _RadauSolveExecutionContext,
    carry0: _RadauAcceptedStepCarry,
    trace: _RadauAdaptiveRolloutTrace,
    *,
    target_attempt_index,
) -> _RadauAcceptedStepZeroTangentComparison:
    """Compare custom vs direct one-step JVP with zero tangent at one attempt."""

    active_mask = np.asarray(jax.device_get(trace.active_mask), dtype=bool)
    accepted_mask = np.asarray(jax.device_get(trace.accepted_mask), dtype=bool)
    attempted_dts = np.asarray(jax.device_get(trace.attempted_dts))
    next_dts = np.asarray(jax.device_get(trace.next_dts))
    next_recent_reject_count = np.asarray(jax.device_get(trace.next_recent_reject_count))
    next_regrowth_cooldown = np.asarray(jax.device_get(trace.next_regrowth_cooldown))
    next_easy_growth_streak = np.asarray(jax.device_get(trace.next_easy_growth_streak))
    next_lagged_response_valid = np.asarray(jax.device_get(trace.next_lagged_response_valid))

    zero = jnp.asarray(0.0, dtype=execution_context.dtype)

    def _tree_all_finite(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        finite = jnp.asarray(True)
        for leaf in leaves:
            leaf_arr = jnp.asarray(leaf)
            if jnp.issubdtype(leaf_arr.dtype, jnp.inexact):
                finite = jnp.logical_and(finite, jnp.all(jnp.isfinite(leaf_arr)))
        return finite

    def _max_abs_scalar(value):
        if value is None:
            return zero
        leaves = jax.tree_util.tree_leaves(value, is_leaf=lambda x: x is None)
        max_abs = zero
        for leaf in leaves:
            if leaf is None:
                continue
            leaf_arr = jnp.asarray(leaf)
            if not jnp.issubdtype(leaf_arr.dtype, jnp.inexact):
                continue
            max_abs = jnp.maximum(max_abs, jnp.max(jnp.abs(leaf_arr)).astype(execution_context.dtype))
        return max_abs

    target_index = int(target_attempt_index)
    if target_index < 0 or target_index >= active_mask.shape[0] or not active_mask[target_index]:
        return _RadauAcceptedStepZeroTangentComparison(
            target_attempt_index=int(target_index),
            target_was_accepted=False,
            trial_dt=float(zero),
            custom_trial_y_max_abs=float(zero),
            custom_stage_history_max_abs=float(zero),
            custom_finite=False,
            direct_trial_y_max_abs=float(zero),
            direct_stage_history_max_abs=float(zero),
            direct_finite=False,
        )

    carry = carry0
    for idx in range(target_index):
        if not active_mask[idx]:
            break
        dt_value = jnp.asarray(attempted_dts[idx], dtype=execution_context.dtype)
        next_dt_value = jnp.asarray(next_dts[idx], dtype=execution_context.dtype)
        carry_for_step = dataclasses.replace(carry, dt=dt_value)
        attempt_result = _execute_radau_accepted_step_attempt(
            execution_context.kernel_context,
            execution_context.physics_context,
            carry_for_step,
            execution_context.attempt_context,
        )
        if accepted_mask[idx]:
            accepted_y = _project_flat_state_if_needed(
                attempt_result.trial_y,
                execution_context.physics_context.project_flat,
            )
            carry = dataclasses.replace(
                carry,
                t=carry.t + dt_value,
                y=accepted_y,
                dt=next_dt_value,
                prev_error=jnp.maximum(
                    attempt_result.err_norm,
                    jnp.asarray(1.0e-12, dtype=execution_context.dtype),
                ),
                prev_stages=attempt_result.stage_history,
                prev_dt=dt_value,
                recent_reject_count=jnp.asarray(next_recent_reject_count[idx]),
                regrowth_cooldown=jnp.asarray(next_regrowth_cooldown[idx]),
                easy_growth_streak=jnp.asarray(next_easy_growth_streak[idx]),
                lagged_response_valid=jnp.asarray(next_lagged_response_valid[idx]),
                jacobian=attempt_result.jacobian_out,
                cache_valid=attempt_result.cache_valid_out,
                cache_dt=attempt_result.cache_dt_out,
                cache_age=attempt_result.cache_age_out,
                real_lu=attempt_result.real_lu_out,
                real_piv=attempt_result.real_piv_out,
                complex_lu=attempt_result.complex_lu_out,
                complex_piv=attempt_result.complex_piv_out,
                prev_theta_final=attempt_result.theta_final,
                prev_newton_iter_count=attempt_result.newton_iter_count,
            )
        else:
            carry = dataclasses.replace(
                carry,
                dt=next_dt_value,
                recent_reject_count=jnp.asarray(next_recent_reject_count[idx]),
                regrowth_cooldown=jnp.asarray(next_regrowth_cooldown[idx]),
                easy_growth_streak=jnp.asarray(next_easy_growth_streak[idx]),
                lagged_response_valid=jnp.asarray(next_lagged_response_valid[idx]),
                jacobian=attempt_result.jacobian_out,
                cache_valid=attempt_result.cache_valid_out,
                cache_dt=attempt_result.cache_dt_out,
                cache_age=attempt_result.cache_age_out,
                real_lu=attempt_result.real_lu_out,
                real_piv=attempt_result.real_piv_out,
                complex_lu=attempt_result.complex_lu_out,
                complex_piv=attempt_result.complex_piv_out,
                prev_theta_final=attempt_result.theta_final,
                prev_newton_iter_count=attempt_result.newton_iter_count,
            )

    target_dt = jnp.asarray(attempted_dts[target_index], dtype=execution_context.dtype)
    carry_for_target = dataclasses.replace(carry, dt=target_dt)
    zero_tangent = jax.tree_util.tree_map(
        lambda x: None if x is None else jnp.zeros_like(x),
        carry_for_target,
        is_leaf=lambda x: x is None,
    )

    def _custom_step(carry_value):
        return _execute_radau_accepted_step_attempt_autodiff(
            execution_context.kernel_context,
            execution_context.physics_context,
            _radau_carry_with_forward_only_jvp_fields(carry_value),
            execution_context.attempt_context,
        )

    def _direct_step(carry_value):
        return _execute_radau_accepted_step_attempt(
            execution_context.kernel_context,
            execution_context.physics_context,
            _radau_carry_with_forward_only_jvp_fields(carry_value),
            execution_context.attempt_context,
        )

    _, custom_tangent = jax.jvp(_custom_step, (carry_for_target,), (zero_tangent,))
    _, direct_tangent = jax.jvp(_direct_step, (carry_for_target,), (zero_tangent,))

    return _RadauAcceptedStepZeroTangentComparison(
        target_attempt_index=int(target_index),
        target_was_accepted=bool(accepted_mask[target_index]),
        trial_dt=float(target_dt),
        custom_trial_y_max_abs=float(_max_abs_scalar(custom_tangent.trial_y)),
        custom_stage_history_max_abs=float(_max_abs_scalar(custom_tangent.stage_history)),
        custom_finite=bool(_tree_all_finite(custom_tangent)),
        direct_trial_y_max_abs=float(_max_abs_scalar(direct_tangent.trial_y)),
        direct_stage_history_max_abs=float(_max_abs_scalar(direct_tangent.stage_history)),
        direct_finite=bool(_tree_all_finite(direct_tangent)),
    )


def _build_prepared_radau_accepted_rollout(
    *,
    solver: "RADAUSolver",
    state,
    vector_field,
    species,
) -> _PreparedRadauAcceptedRollout:
    """Build reusable Radau accepted-rollout contexts without running solve.

    This is the setup boundary shared by the production Radau solve and the
    separate AD-facing fixed-`dt` accepted-rollout helper.
    """
    args = (species,)
    kwargs: dict[str, Any] = {}
    temperature_active_mask, fixed_temperature_profile = _extract_fixed_temperature_projection(vector_field)
    density_floor, temperature_floor = _extract_state_regularization(vector_field)
    flat_state0, unpack_flat, _unpack_packed, _pack_state, project_flat = _make_solver_state_transform(
        state,
        species,
        temperature_active_mask=temperature_active_mask,
        fixed_temperature_profile=fixed_temperature_profile,
        density_floor=density_floor,
        temperature_floor=temperature_floor,
    )
    dtype = flat_state0.dtype
    density0 = getattr(state, "density", None)
    pressure0 = getattr(state, "pressure", None)
    er0 = getattr(state, "Er", None)
    if density0 is not None and pressure0 is not None and er0 is not None:
        density_size = int(np.prod(density0.shape))
        pressure_size = int(np.prod(pressure0.shape))
        er_size = int(np.prod(er0.shape))
    else:
        density_size = 0
        pressure_size = 0
        er_size = 0
    if solver.num_stages not in _RADAU_STAGE_CONFIGS:
        raise ValueError(
            f"Unsupported custom Radau stage count '{solver.num_stages}'. "
            f"Available options: {sorted(_RADAU_STAGE_CONFIGS)}"
        )
    stage_cfg = _RADAU_STAGE_CONFIGS[solver.num_stages]
    c = jnp.asarray(stage_cfg.c, dtype=dtype)
    a = jnp.asarray(stage_cfg.a, dtype=dtype)
    b = jnp.asarray(stage_cfg.b, dtype=dtype)
    b_error = jnp.asarray(stage_cfg.b_error, dtype=dtype)
    embedded_f0_weight = jnp.asarray(stage_cfg.embedded_f0_weight, dtype=dtype)
    num_stages = int(stage_cfg.num_stages)
    t0 = jnp.asarray(solver.t0, dtype=dtype)
    t_final = jnp.asarray(solver.t1, dtype=dtype)
    dt_min = jnp.asarray(solver.min_step, dtype=dtype)
    dt_max = jnp.asarray(solver.max_step, dtype=dtype)
    base_dt = jnp.clip(jnp.asarray(solver.dt, dtype=dtype), dt_min, dt_max)
    state_dim = flat_state0.shape[0]
    rhs_mode = str(getattr(solver, "rhs_mode", "black_box")).strip().lower()
    error_scale_mode = (
        str(getattr(solver, "error_estimator", "embedded2")).strip().lower().removeprefix("embedded2_")
    )
    if error_scale_mode == "embedded2":
        error_scale_mode = "max"
    elif error_scale_mode in {"mean_scale", "mean"}:
        error_scale_mode = "mean"
    elif error_scale_mode in {"blend_scale", "blend"}:
        error_scale_mode = "blend"
    elif error_scale_mode == "ntss_scale":
        error_scale_mode = "ntss"
    elif error_scale_mode == "ntss_max_scale":
        error_scale_mode = "ntss_max"
    elif error_scale_mode == "ntss_blend_scale":
        error_scale_mode = "ntss_blend"
    elif error_scale_mode == "ntss_transport_scale":
        error_scale_mode = "ntss_transport"
    elif error_scale_mode == "ntss_block_floor_scale":
        error_scale_mode = "ntss_block_floor"
    elif error_scale_mode == "ntss_block_rms":
        error_scale_mode = "ntss_block_rms"
    else:
        error_scale_mode = "max"

    flat_rhs = _flat_rhs_factory(unpack_flat, vector_field, args, kwargs, project_flat=project_flat)
    build_lagged_response_raw, _ = _lagged_response_hooks(vector_field)
    flat_rhs_with_lagged_response_raw = _flat_rhs_with_lagged_response_factory(
        unravel=unpack_flat,
        vector_field=vector_field,
        args=args,
        kwargs=kwargs,
        project_flat=project_flat,
    )
    use_transport_lagged_response = rhs_mode in {"lagged_transport_response", "lagged_response"}
    if use_transport_lagged_response and build_lagged_response_raw is None:
        raise ValueError(
            "Radau lagged transport response mode requires a vector field owner that defines "
            "build_lagged_response(...) and evaluate_with_lagged_response(...)."
        )
    build_lagged_response = build_lagged_response_raw
    flat_rhs_with_lagged_response = flat_rhs_with_lagged_response_raw
    initial_lagged_response = (
        build_lagged_response(unpack_flat(_project_flat_state_if_needed(flat_state0, project_flat)))
        if (use_transport_lagged_response and build_lagged_response is not None)
        else None
    )
    initial_rhs = (
        flat_rhs_with_lagged_response(t0, flat_state0, initial_lagged_response)
        if initial_lagged_response is not None
        else flat_rhs(t0, flat_state0)
    )

    predictor_mode = str(getattr(solver, "predictor_mode", "current")).strip().lower()
    divergence_mode = str(getattr(solver, "newton_divergence_mode", "legacy")).strip().lower()
    residual_norm_mode = str(getattr(solver, "newton_residual_norm", "raw")).strip().lower()
    debug_newton_trace = bool(getattr(solver, "debug_stage_markers", False))
    use_rms_residual_norm = residual_norm_mode in {"rms", "scaled_rms"}
    conservative_divergence = divergence_mode in {"ntss", "hairer", "conservative"}
    newton_tol_mode = str(getattr(solver, "newton_tol_mode", "residual")).strip().lower()
    fnewt_mode = str(getattr(solver, "newton_fnewt_mode", "tol")).strip().lower()
    use_hairer_newton_tol = newton_tol_mode in {"hairer", "hairer_like", "ntss"}
    use_hairer_scaled_correction = use_hairer_newton_tol or fnewt_mode in {"hairer", "hairer_like", "ntss"}
    tiny_scalar = jnp.asarray(1.0e-30, dtype=dtype)
    zero_scalar = jnp.asarray(0.0, dtype=dtype)
    theta_diverge_threshold = jnp.asarray(0.99 if conservative_divergence else 0.99, dtype=dtype)
    predictor_defect_floor = jnp.asarray(1.0e-4, dtype=dtype)
    predictor_defect_cap = jnp.asarray(20.0, dtype=dtype)
    residual_blowup_factor = jnp.asarray(2.0, dtype=dtype)
    theta_clip_min = jnp.asarray(0.1, dtype=dtype)
    theta_clip_max = jnp.asarray(1.5, dtype=dtype)
    newton_shrink_num = jnp.asarray(0.8, dtype=dtype)
    newton_shrink_min = jnp.asarray(0.1, dtype=dtype)
    newton_shrink_max = jnp.asarray(0.5, dtype=dtype)
    if fnewt_mode in {"hairer", "hairer_like", "ntss"}:
        uround = jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)
        expmns = jnp.asarray((num_stages + 1.0) / (2.0 * num_stages), dtype=dtype)
        safe_rtol = jnp.maximum(jnp.asarray(solver.rtol, dtype=dtype), uround * 10.0)
        rtol1 = jnp.asarray(0.1, dtype=dtype) * (safe_rtol ** expmns)
        expmi = jnp.asarray(1.0, dtype=dtype) / expmns
        predictor_fnewt = jnp.maximum(
            jnp.asarray(10.0, dtype=dtype) * uround / rtol1,
            jnp.minimum(
                jnp.asarray(3.0e-2, dtype=dtype),
                rtol1 ** (expmi - jnp.asarray(1.0, dtype=dtype)),
            ),
        )
    else:
        predictor_fnewt = jnp.maximum(jnp.asarray(solver.tol, dtype=dtype), tiny_scalar)
    estimator_rtol_eff = jnp.asarray(solver.rtol, dtype=dtype)
    if error_scale_mode == "ntss":
        uround_est = jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)
        expmns_est = jnp.asarray((num_stages + 1.0) / (2.0 * num_stages), dtype=dtype)
        safe_rtol_est = jnp.maximum(jnp.asarray(solver.rtol, dtype=dtype), uround_est * 10.0)
        estimator_rtol_eff = jnp.asarray(0.1, dtype=dtype) * (safe_rtol_est ** expmns_est)
    if error_scale_mode in {"ntss_max", "ntss_blend", "ntss_transport", "ntss_block_floor", "ntss_block_rms"}:
        uround_est = jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)
        expmns_est = jnp.asarray((num_stages + 1.0) / (2.0 * num_stages), dtype=dtype)
        safe_rtol_est = jnp.maximum(jnp.asarray(solver.rtol, dtype=dtype), uround_est * 10.0)
        estimator_rtol_eff = jnp.asarray(0.1, dtype=dtype) * (safe_rtol_est ** expmns_est)

    radau_transform = jnp.asarray(stage_cfg.transform, dtype=dtype)
    radau_inv_transform = jnp.asarray(stage_cfg.inv_transform, dtype=dtype)
    radau_real_eig = jnp.asarray(stage_cfg.real_eig, dtype=dtype)
    radau_complex_blocks = jnp.asarray(stage_cfg.complex_blocks, dtype=dtype)
    num_complex_pairs = int(stage_cfg.complex_blocks.shape[0])
    identity_2 = jnp.eye(2, dtype=dtype)
    identity_n = jnp.eye(state_dim, dtype=dtype)
    real_lu0 = jnp.eye(state_dim, dtype=dtype)
    real_piv0 = jnp.arange(state_dim, dtype=jnp.int32)
    complex_dim = 2 * state_dim
    complex_lu0 = jnp.broadcast_to(jnp.eye(complex_dim, dtype=dtype), (num_complex_pairs, complex_dim, complex_dim))
    complex_piv0 = jnp.broadcast_to(jnp.arange(complex_dim, dtype=jnp.int32), (num_complex_pairs, complex_dim))
    residual_size_sqrt = jnp.sqrt(jnp.asarray(num_stages * state_dim, dtype=dtype))
    state_scale_base = jnp.asarray(solver.atol, dtype=dtype) + jnp.asarray(solver.rtol, dtype=dtype) * jnp.abs(flat_state0)
    stage_scale = jnp.broadcast_to(jnp.maximum(state_scale_base, tiny_scalar)[None, :], (num_stages, state_dim)).reshape((-1,))

    kernel_context = _RadauAcceptedStepKernelContext(
        radau_transform=radau_transform,
        radau_inv_transform=radau_inv_transform,
        c=c,
        a=a,
        b=b,
        b_error=b_error,
        embedded_f0_weight=embedded_f0_weight,
        dtype=dtype,
        density_size=int(density_size),
        pressure_size=int(pressure_size),
        er_size=int(er_size),
        predictor_mode=predictor_mode,
        jacobian_reuse_rtol=jnp.asarray(solver.jacobian_reuse_rtol, dtype=dtype),
        use_lagged_linear_response=bool(rhs_mode == "lagged_linear_state"),
        num_stages=int(num_stages),
        state_dim=int(state_dim),
        num_complex_pairs=int(num_complex_pairs),
        complex_dim=int(complex_dim),
        identity_2=identity_2,
        identity_n=identity_n,
        use_rms_residual_norm=bool(use_rms_residual_norm),
        residual_size_sqrt=residual_size_sqrt,
        stage_scale=stage_scale,
        radau_real_eig=radau_real_eig,
        radau_complex_blocks=radau_complex_blocks,
        predictor_fnewt=predictor_fnewt,
        maxiter=int(solver.maxiter),
        tol=jnp.asarray(solver.tol, dtype=dtype),
        error_scale_mode=error_scale_mode,
        atol=jnp.asarray(solver.atol, dtype=dtype),
        rtol=jnp.asarray(solver.rtol, dtype=dtype),
        estimator_rtol_eff=estimator_rtol_eff,
        use_hairer_newton_tol=bool(use_hairer_newton_tol),
        use_hairer_scaled_correction=bool(use_hairer_scaled_correction),
        theta_diverge_threshold=theta_diverge_threshold,
        predictor_defect_floor=predictor_defect_floor,
        predictor_defect_cap=predictor_defect_cap,
        residual_blowup_factor=residual_blowup_factor,
        theta_clip_min=theta_clip_min,
        theta_clip_max=theta_clip_max,
        newton_shrink_num=newton_shrink_num,
        newton_shrink_min=newton_shrink_min,
        newton_shrink_max=newton_shrink_max,
        tiny_scalar=tiny_scalar,
        zero_scalar=zero_scalar,
        debug_newton_trace=bool(debug_newton_trace),
        use_transport_lagged_response=bool(use_transport_lagged_response),
    )
    physics_context = _RadauAcceptedStepPhysicsContext(
        unpack_flat=unpack_flat,
        project_flat=project_flat,
        build_lagged_response=build_lagged_response,
        flat_rhs=flat_rhs,
        flat_rhs_with_lagged_response=flat_rhs_with_lagged_response,
    )
    step_state0 = _make_radau_initial_step_state(
        t0,
        flat_state0,
        base_dt,
        dtype,
        initial_rhs,
        num_stages,
        real_lu0,
        real_piv0,
        complex_lu0,
        complex_piv0,
        initial_lagged_response,
        jnp.asarray(use_transport_lagged_response),
        flat_state0,
    )
    initial_carry = _radau_carry_from_step_state(step_state0)
    return _PreparedRadauAcceptedRollout(
        kernel_context=kernel_context,
        physics_context=physics_context,
        initial_carry=initial_carry,
    )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class _RadauStepInfo:
    y: Any
    t: Any
    dt: Any
    next_dt: Any = None
    growth: Any = None
    lagged_reused: Any = None
    jacobian_reused: Any = None
    accepted: Any = None
    failed: Any = None
    fail_code: Any = None
    converged: Any = None
    err_norm: Any = None
    diverged: Any = None
    nonfinite_stage_state: Any = None
    nonfinite_stage_residual: Any = None
    finite_f0: Any = None
    finite_z0: Any = None
    finite_initial_residual: Any = None
    newton_iter_count: Any = None
    final_residual_norm: Any = None
    final_delta_norm: Any = None
    theta_final: Any = None
    slow_contraction: Any = None
    residual_blowup: Any = None
    newton_nonfinite: Any = None


class RADAUSolver(_RadauSolverConfig):
    def solve(self, state, vector_field: Callable, *args, **kwargs):
        STATUS_FAILED = 0
        STATUS_FAIL_CODE = 1
        STATUS_N_ACCEPTED = 2
        species = _extract_species_from_args(args)
        temperature_active_mask, fixed_temperature_profile = _extract_fixed_temperature_projection(vector_field)
        density_floor, temperature_floor = _extract_state_regularization(vector_field)
        state = _project_state_to_quasi_neutrality(
            state,
            species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
            density_floor=density_floor,
            temperature_floor=temperature_floor,
        )
        flat_state0, unpack_flat, _unpack_packed, _pack_state, project_flat = _make_solver_state_transform(
            state,
            species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
            density_floor=density_floor,
            temperature_floor=temperature_floor,
        )
        packed_state0 = _pack_transport_state_arrays(state, species)
        if isinstance(packed_state0, tuple) and len(packed_state0) == 3:
            density0, pressure0, er0 = packed_state0
            density_size = int(np.prod(density0.shape))
            pressure_size = int(np.prod(pressure0.shape))
            er_size = int(np.prod(er0.shape))
        else:
            density_size = 0
            pressure_size = 0
            er_size = 0
        if self.num_stages not in _RADAU_STAGE_CONFIGS:
            raise ValueError(
                f"Unsupported custom Radau stage count '{self.num_stages}'. "
                f"Available stage configurations: {sorted(_RADAU_STAGE_CONFIGS)}."
            )
        stage_cfg = _RADAU_STAGE_CONFIGS[self.num_stages]
        dtype = flat_state0.dtype
        c = jnp.asarray(stage_cfg.c, dtype=dtype)
        a = jnp.asarray(stage_cfg.a, dtype=dtype)
        b = jnp.asarray(stage_cfg.b, dtype=dtype)
        b_error = jnp.asarray(stage_cfg.b_error, dtype=dtype)
        embedded_f0_weight = jnp.asarray(stage_cfg.embedded_f0_weight, dtype=dtype)
        num_stages = int(stage_cfg.num_stages)
        t0 = jnp.asarray(self.t0, dtype=dtype)
        t_final = jnp.asarray(self.t1, dtype=dtype)
        dt_min = jnp.asarray(self.min_step, dtype=dtype)
        dt_max = jnp.asarray(self.max_step, dtype=dtype)
        base_dt = jnp.clip(jnp.asarray(self.dt, dtype=dtype), dt_min, dt_max)
        max_total_steps = int(max(1, self.max_steps))
        state_dim = flat_state0.shape[0]
        error_estimator_mode = str(getattr(self, "error_estimator", "embedded2")).strip().lower()
        error_scale_mode = (
            "mean" if error_estimator_mode == "embedded2_mean_scale"
            else "blend" if error_estimator_mode == "embedded2_blend_scale"
            else "ntss" if error_estimator_mode == "embedded2_ntss_scale"
            else "ntss_max" if error_estimator_mode == "embedded2_ntss_max_scale"
            else "ntss_blend" if error_estimator_mode == "embedded2_ntss_blend_scale"
            else "ntss_transport" if error_estimator_mode == "embedded2_ntss_transport_scale"
            else "ntss_block_floor" if error_estimator_mode == "embedded2_ntss_block_floor_scale"
            else "ntss_block_rms" if error_estimator_mode == "embedded2_ntss_block_rms"
            else "max"
        )
        flat_rhs = _flat_rhs_factory(unpack_flat, vector_field, args, kwargs, project_flat=project_flat)
        build_lagged_response_raw, _ = _lagged_response_hooks(vector_field)
        flat_rhs_with_lagged_response_raw = _flat_rhs_with_lagged_response_factory(
            unravel=unpack_flat,
            vector_field=vector_field,
            args=args,
            kwargs=kwargs,
            project_flat=project_flat,
        )
        rhs_mode = str(getattr(self, "rhs_mode", "black_box")).strip().lower()
        if rhs_mode not in {"black_box", "lagged_linear_state", "lagged_transport_response", "lagged_response"}:
            raise ValueError(
                f"Unsupported Radau rhs_mode '{rhs_mode}'. "
                "Expected one of ['black_box', 'lagged_linear_state', 'lagged_transport_response', 'lagged_response']."
            )
        use_lagged_linear_response = rhs_mode == "lagged_linear_state"
        use_transport_lagged_response = rhs_mode in {"lagged_transport_response", "lagged_response"}
        if use_transport_lagged_response and build_lagged_response_raw is None:
            raise ValueError(
                "Radau lagged transport response mode requires a vector field with "
                "build_lagged_response(...) and evaluate_with_lagged_response(...)."
            )
        build_lagged_response = build_lagged_response_raw
        flat_rhs_with_lagged_response = flat_rhs_with_lagged_response_raw
        initial_lagged_response = (
            build_lagged_response(unpack_flat(_project_flat_state_if_needed(flat_state0, project_flat)))
            if (use_transport_lagged_response and build_lagged_response is not None)
            else None
        )
        initial_rhs = (
            flat_rhs_with_lagged_response(t0, flat_state0, initial_lagged_response)
            if initial_lagged_response is not None
            else flat_rhs(t0, flat_state0)
        )
        predictor_mode = str(getattr(self, "predictor_mode", "current")).strip().lower()
        error_order = float(stage_cfg.embedded_order if stage_cfg.has_embedded_estimator else stage_cfg.order)
        controller_alpha = 0.7 / (error_order + 1.0)
        zero_scalar = jnp.asarray(0.0, dtype=dtype)
        tiny_scalar = jnp.asarray(1.0e-30, dtype=dtype)
        divergence_mode = str(getattr(self, "newton_divergence_mode", "legacy")).strip().lower()
        residual_norm_mode = str(getattr(self, "newton_residual_norm", "raw")).strip().lower()
        debug_newton_trace = bool(getattr(self, "debug_stage_markers", False))
        conservative_divergence = divergence_mode in {"conservative", "hairer_like", "hairer"}
        use_rms_residual_norm = residual_norm_mode in {"rms", "scaled", "normalized"}
        newton_tol_mode = str(getattr(self, "newton_tol_mode", "residual")).strip().lower()
        fnewt_mode = str(getattr(self, "newton_fnewt_mode", "tol")).strip().lower()
        use_hairer_newton_tol = newton_tol_mode in {"hairer", "hairer_like", "ntss"}
        use_hairer_scaled_correction = use_hairer_newton_tol or fnewt_mode in {"hairer", "hairer_like", "ntss"}
        # Follow the Hairer/NTSS pattern more closely: use the Newton-correction
        # contraction estimate to predict the remaining defect, and turn that
        # into a controlled step rejection rather than an immediate "nonfinite"
        # style divergence classification.
        theta_diverge_threshold = jnp.asarray(0.99 if conservative_divergence else 0.99, dtype=dtype)
        predictor_defect_floor = jnp.asarray(1.0e-4, dtype=dtype)
        predictor_defect_cap = jnp.asarray(20.0, dtype=dtype)
        residual_blowup_factor = jnp.asarray(2.0, dtype=dtype)
        theta_clip_min = jnp.asarray(0.1, dtype=dtype)
        theta_clip_max = jnp.asarray(1.5, dtype=dtype)
        newton_shrink_num = jnp.asarray(0.8, dtype=dtype)
        newton_shrink_min = jnp.asarray(0.1, dtype=dtype)
        newton_shrink_max = jnp.asarray(0.5, dtype=dtype)
        if fnewt_mode in {"hairer", "hairer_like", "ntss"}:
            uround = jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)
            expmns = jnp.asarray((num_stages + 1.0) / (2.0 * num_stages), dtype=dtype)
            safe_rtol = jnp.maximum(jnp.asarray(self.rtol, dtype=dtype), uround * 10.0)
            rtol1 = jnp.asarray(0.1, dtype=dtype) * (safe_rtol ** expmns)
            expmi = jnp.asarray(1.0, dtype=dtype) / expmns
            predictor_fnewt = jnp.maximum(
                jnp.asarray(10.0, dtype=dtype) * uround / rtol1,
                jnp.minimum(
                    jnp.asarray(3.0e-2, dtype=dtype),
                    rtol1 ** (expmi - jnp.asarray(1.0, dtype=dtype)),
                ),
            )
        else:
            predictor_fnewt = jnp.maximum(jnp.asarray(self.tol, dtype=dtype), tiny_scalar)
        estimator_rtol_eff = jnp.asarray(self.rtol, dtype=dtype)
        if error_estimator_mode == "embedded2_ntss_scale":
            uround_est = jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)
            expmns_est = jnp.asarray((num_stages + 1.0) / (2.0 * num_stages), dtype=dtype)
            safe_rtol_est = jnp.maximum(jnp.asarray(self.rtol, dtype=dtype), uround_est * 10.0)
            estimator_rtol_eff = jnp.asarray(0.1, dtype=dtype) * (safe_rtol_est ** expmns_est)
        if error_estimator_mode in {"embedded2_ntss_max_scale", "embedded2_ntss_blend_scale", "embedded2_ntss_transport_scale", "embedded2_ntss_block_floor_scale", "embedded2_ntss_block_rms"}:
            uround_est = jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)
            expmns_est = jnp.asarray((num_stages + 1.0) / (2.0 * num_stages), dtype=dtype)
            safe_rtol_est = jnp.maximum(jnp.asarray(self.rtol, dtype=dtype), uround_est * 10.0)
            estimator_rtol_eff = jnp.asarray(0.1, dtype=dtype) * (safe_rtol_est ** expmns_est)
        radau_transform = jnp.asarray(stage_cfg.transform, dtype=dtype)
        radau_inv_transform = jnp.asarray(stage_cfg.inv_transform, dtype=dtype)
        radau_real_eig = jnp.asarray(stage_cfg.real_eig, dtype=dtype)
        radau_complex_blocks = jnp.asarray(stage_cfg.complex_blocks, dtype=dtype)
        num_complex_pairs = int(stage_cfg.complex_blocks.shape[0])
        identity_2 = jnp.eye(2, dtype=dtype)
        identity_n = jnp.eye(state_dim, dtype=dtype)
        real_lu0 = jnp.eye(state_dim, dtype=dtype)
        real_piv0 = jnp.arange(state_dim, dtype=jnp.int32)
        complex_dim = 2 * state_dim
        complex_lu0 = jnp.broadcast_to(jnp.eye(complex_dim, dtype=dtype), (num_complex_pairs, complex_dim, complex_dim))
        complex_piv0 = jnp.broadcast_to(jnp.arange(complex_dim, dtype=jnp.int32), (num_complex_pairs, complex_dim))
        residual_size_sqrt = jnp.sqrt(jnp.asarray(num_stages * state_dim, dtype=dtype))

        if debug_newton_trace:
            jax.debug.print(
                "[radau-solver] newton_tol_mode={tol_mode} newton_fnewt_mode={fnewt_mode} predictor_fnewt={fnewt:.6e} residual_tol={tol:.6e}",
                tol_mode=newton_tol_mode,
                fnewt_mode=fnewt_mode,
                tol=jnp.asarray(self.tol, dtype=dtype),
                fnewt=predictor_fnewt,
            )

        state_scale_base = jnp.asarray(self.atol, dtype=dtype) + jnp.asarray(self.rtol, dtype=dtype) * jnp.abs(flat_state0)
        stage_scale = jnp.broadcast_to(jnp.maximum(state_scale_base, tiny_scalar)[None, :], (num_stages, state_dim)).reshape((-1,))

        kernel_context = _RadauAcceptedStepKernelContext(
            radau_transform=radau_transform,
            radau_inv_transform=radau_inv_transform,
            c=c,
            a=a,
            b=b,
            b_error=b_error,
            embedded_f0_weight=embedded_f0_weight,
            dtype=dtype,
            density_size=int(density_size),
            pressure_size=int(pressure_size),
            er_size=int(er_size),
            predictor_mode=predictor_mode,
            jacobian_reuse_rtol=jnp.asarray(self.jacobian_reuse_rtol, dtype=dtype),
            use_lagged_linear_response=bool(use_lagged_linear_response),
            num_stages=int(num_stages),
            state_dim=int(state_dim),
            num_complex_pairs=int(num_complex_pairs),
            complex_dim=int(complex_dim),
            identity_2=identity_2,
            identity_n=identity_n,
            use_rms_residual_norm=bool(use_rms_residual_norm),
            residual_size_sqrt=residual_size_sqrt,
            stage_scale=stage_scale,
            radau_real_eig=radau_real_eig,
            radau_complex_blocks=radau_complex_blocks,
            predictor_fnewt=predictor_fnewt,
            maxiter=int(self.maxiter),
            tol=jnp.asarray(self.tol, dtype=dtype),
            error_scale_mode=error_scale_mode,
            atol=jnp.asarray(self.atol, dtype=dtype),
            rtol=jnp.asarray(self.rtol, dtype=dtype),
            estimator_rtol_eff=estimator_rtol_eff,
            use_hairer_newton_tol=bool(use_hairer_newton_tol),
            use_hairer_scaled_correction=bool(use_hairer_scaled_correction),
            theta_diverge_threshold=theta_diverge_threshold,
            predictor_defect_floor=predictor_defect_floor,
            predictor_defect_cap=predictor_defect_cap,
            residual_blowup_factor=residual_blowup_factor,
            theta_clip_min=theta_clip_min,
            theta_clip_max=theta_clip_max,
            newton_shrink_num=newton_shrink_num,
            newton_shrink_min=newton_shrink_min,
            newton_shrink_max=newton_shrink_max,
            tiny_scalar=tiny_scalar,
            zero_scalar=zero_scalar,
            debug_newton_trace=bool(debug_newton_trace),
            use_transport_lagged_response=bool(use_transport_lagged_response),
        )
        physics_context = _RadauAcceptedStepPhysicsContext(
            unpack_flat=unpack_flat,
            project_flat=project_flat,
            build_lagged_response=build_lagged_response,
            flat_rhs=flat_rhs,
            flat_rhs_with_lagged_response=flat_rhs_with_lagged_response,
        )

        attempt_context = _RadauAcceptedStepAttemptContext(
            t_final=t_final,
            use_transport_lagged_response=jnp.asarray(use_transport_lagged_response),
        )
        execution_context = _RadauSolveExecutionContext(
            kernel_context=kernel_context,
            physics_context=physics_context,
            attempt_context=attempt_context,
            dtype=dtype,
            dt_min=dt_min,
            dt_max=dt_max,
            safety_factor=self.safety_factor,
            controller_alpha=controller_alpha,
            min_step_factor=self.min_step_factor,
            max_step_factor=self.max_step_factor,
            controller_mode=str(getattr(self, "controller_mode", "current")).strip().lower(),
            use_transport_lagged_response=use_transport_lagged_response,
            lagged_response_reuse_mode=str(getattr(self, "lagged_response_reuse_mode", "retry_only")).strip().lower(),
            lagged_response_reuse_rtol=jnp.asarray(getattr(self, "lagged_response_reuse_rtol", 5.0e-2), dtype=dtype),
            lagged_response_reuse_atol=jnp.asarray(getattr(self, "lagged_response_reuse_atol", 1.0e-8), dtype=dtype),
            project_flat=project_flat,
            debug_newton_trace=bool(debug_newton_trace),
        )

        step_fn = partial(_radau_step_fn, execution_context)

        step_state0 = _make_radau_initial_step_state(
            t0,
            flat_state0,
            base_dt,
            dtype,
            initial_rhs,
            num_stages,
            real_lu0,
            real_piv0,
            complex_lu0,
            complex_piv0,
            initial_lagged_response,
            jnp.asarray(use_transport_lagged_response),
            flat_state0,
        )
        save_n = getattr(self, "save_n", None)
        save_n = max(1, int(save_n)) if save_n is not None else 1
        stop_after_accepted_steps = getattr(self, "stop_after_accepted_steps", None)
        if bool(getattr(self, "debug_walltime_attempts", False)):
            loop_result = _run_saved_loop_debug_walltime(
                step_state0=step_state0,
                step_fn=step_fn,
                save_n=save_n,
                t0=t0,
                t_final=t_final,
                state_dim=state_dim,
                dtype=dtype,
                max_total_steps=max_total_steps,
                stop_after_accepted_steps=stop_after_accepted_steps,
                walltime_label="radau.attempt",
            )
        else:
            loop_result = _run_saved_loop(
                step_state0=step_state0,
                step_fn=step_fn,
                save_n=save_n,
                t0=t0,
                t_final=t_final,
                state_dim=state_dim,
                dtype=dtype,
                max_total_steps=max_total_steps,
                stop_after_accepted_steps=stop_after_accepted_steps,
            )
        (
            step_state_f,
            _,
            _,
            ys_saved,
            ts_saved,
            dts_saved,
            accepted_mask_saved,
            failed_mask_saved,
            fail_codes_saved,
            last_attempt_accepted,
            last_attempt_converged,
            last_attempt_err_norm,
            last_attempt_fail_code,
            last_attempt_diverged,
            last_attempt_nonfinite_stage_state,
            last_attempt_nonfinite_stage_residual,
            last_attempt_finite_f0,
            last_attempt_finite_z0,
            last_attempt_finite_initial_residual,
            last_attempt_newton_iter_count,
            last_attempt_final_residual_norm,
            last_attempt_final_delta_norm,
            last_attempt_theta_final,
            last_attempt_slow_contraction,
            last_attempt_residual_blowup,
            last_attempt_newton_nonfinite,
            last_attempt_lagged_reused,
            last_attempt_jacobian_reused,
        ) = loop_result
        failed_f = step_state_f.status[STATUS_FAILED] != 0
        fail_code_f = step_state_f.status[STATUS_FAIL_CODE]
        n_acc_f = step_state_f.status[STATUS_N_ACCEPTED]
        accepted_limit_hit = _accepted_step_limit_reached(step_state_f, stop_after_accepted_steps)
        return _finalize_custom_solver_output(
            ys_saved, ts_saved, dts_saved, accepted_mask_saved, failed_mask_saved, fail_codes_saved,
            step_state_f.y,
            step_state_f.t,
            jnp.logical_or(step_state_f.t >= (t_final - 1.0e-15), accepted_limit_hit),
            failed_f,
            fail_code_f,
            n_acc_f,
            last_attempt_accepted,
            last_attempt_converged,
            last_attempt_err_norm,
            last_attempt_fail_code,
            last_attempt_diverged,
            last_attempt_nonfinite_stage_state,
            last_attempt_nonfinite_stage_residual,
            last_attempt_finite_f0,
            last_attempt_finite_z0,
            last_attempt_finite_initial_residual,
            last_attempt_newton_iter_count,
            last_attempt_final_residual_norm,
            last_attempt_final_delta_norm,
            last_attempt_theta_final,
            last_attempt_slow_contraction,
            last_attempt_residual_blowup,
            last_attempt_newton_nonfinite,
            last_attempt_lagged_reused,
            last_attempt_jacobian_reused,
            getattr(step_state_f, "reuse_state", None),
            unpack_flat,
            state,
            species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
            density_floor=density_floor,
            temperature_floor=temperature_floor,
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class _ThetaSolverConfig(TransportSolver):
    t0: float
    t1: float
    dt: float
    min_step: float = 1.0e-14
    theta_implicit: float = 1.0
    predictor_mode: str = "linearized"
    rhs_mode: str = "black_box"
    use_predictor_corrector: bool = False
    n_corrector_steps: int = 1
    tol: float = 1.0e-8
    max_steps: int = 20000
    stop_after_accepted_steps: int | None = None
    n_steps: int = 0

    def __init__(
        self,
        t0: float = 0.0,
        t1: float = 1.0,
        dt: float = 1.0e-2,
        min_step: float = 1.0e-14,
        theta_implicit: float = 1.0,
        predictor_mode: str = "linearized",
        rhs_mode: str = "black_box",
        use_predictor_corrector: bool = False,
        n_corrector_steps: int = 1,
        tol: float = 1.0e-8,
        max_steps: int = 20000,
        stop_after_accepted_steps: int | None = None,
        save_n=None,
    ):
        n_steps = max(1, int(jnp.ceil((float(t1) - float(t0)) / float(dt))))
        object.__setattr__(self, "t0", float(t0))
        object.__setattr__(self, "t1", float(t1))
        object.__setattr__(self, "dt", float(dt))
        object.__setattr__(self, "min_step", float(min_step))
        object.__setattr__(self, "theta_implicit", float(theta_implicit))
        predictor_mode_norm = str(predictor_mode).strip().lower()
        if predictor_mode_norm not in {"linearized", "euler"}:
            raise ValueError(
                f"Unsupported theta predictor_mode '{predictor_mode}'. "
                "Expected 'linearized' or 'euler'."
            )
        rhs_mode_norm = str(rhs_mode).strip().lower()
        if rhs_mode_norm not in {"black_box", "lagged_linear_state", "lagged_transport_response", "lagged_response"}:
            raise ValueError(
                f"Unsupported theta rhs_mode '{rhs_mode}'. "
                "Expected one of ['black_box', 'lagged_linear_state', 'lagged_transport_response', 'lagged_response']."
            )
        object.__setattr__(self, "predictor_mode", predictor_mode_norm)
        object.__setattr__(self, "rhs_mode", rhs_mode_norm)
        object.__setattr__(self, "use_predictor_corrector", bool(use_predictor_corrector))
        object.__setattr__(self, "n_corrector_steps", int(max(0, n_corrector_steps)))
        object.__setattr__(self, "tol", float(tol))
        object.__setattr__(self, "max_steps", int(max(1, max_steps)))
        if stop_after_accepted_steps is not None:
            stop_after_accepted_steps = int(max(1, stop_after_accepted_steps))
        object.__setattr__(self, "stop_after_accepted_steps", stop_after_accepted_steps)
        object.__setattr__(self, "n_steps", n_steps)
        object.__setattr__(self, "save_n", save_n)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class _ThetaNewtonSolverConfig(_ThetaSolverConfig):
    maxiter: int = 20
    max_step: float = 1.0
    safety_factor: float = 0.9
    min_step_factor: float = 0.5
    max_step_factor: float = 2.0
    controller_mode: str = "current"
    jacobian_reuse_mode: str = "refresh_each_iteration"
    jacobian_reuse_rtol: float = 0.1
    max_jacobian_age: int = 8
    lagged_response_reuse_mode: str = "retry_only"
    lagged_response_reuse_rtol: float = 5.0e-2
    lagged_response_reuse_atol: float = 1.0e-8
    target_nonlinear_iterations: int = 4
    delta_reduction_factor: float = 0.5
    tau_min: float = 0.01

    def __init__(
        self,
        t0: float = 0.0,
        t1: float = 1.0,
        dt: float = 1.0e-2,
        min_step: float = 1.0e-14,
        theta_implicit: float = 1.0,
        predictor_mode: str = "linearized",
        rhs_mode: str = "black_box",
        use_predictor_corrector: bool = False,
        n_corrector_steps: int = 1,
        tol: float = 1.0e-8,
        maxiter: int = 20,
        max_step: float | None = None,
        safety_factor: float = 0.9,
        min_step_factor: float = 0.5,
        max_step_factor: float = 2.0,
        controller_mode: str = "current",
        jacobian_reuse_mode: str = "refresh_each_iteration",
        jacobian_reuse_rtol: float = 0.1,
        max_jacobian_age: int = 8,
        lagged_response_reuse_mode: str = "retry_only",
        lagged_response_reuse_rtol: float = 5.0e-2,
        lagged_response_reuse_atol: float = 1.0e-8,
        target_nonlinear_iterations: int = 4,
        delta_reduction_factor: float = 0.5,
        tau_min: float = 0.01,
        max_steps: int = 20000,
        stop_after_accepted_steps: int | None = None,
        save_n=None,
    ):
        super().__init__(
            t0=t0,
            t1=t1,
            dt=dt,
            min_step=min_step,
            theta_implicit=theta_implicit,
            predictor_mode=predictor_mode,
            rhs_mode=rhs_mode,
            use_predictor_corrector=use_predictor_corrector,
            n_corrector_steps=n_corrector_steps,
            tol=tol,
            max_steps=max_steps,
            stop_after_accepted_steps=stop_after_accepted_steps,
            save_n=save_n,
        )
        object.__setattr__(self, "maxiter", int(max(1, maxiter)))
        if max_step is None:
            max_step = max(float(t1) - float(t0), float(dt))
        object.__setattr__(self, "max_step", float(max_step))
        object.__setattr__(self, "safety_factor", float(safety_factor))
        object.__setattr__(self, "min_step_factor", float(min_step_factor))
        object.__setattr__(self, "max_step_factor", float(max_step_factor))
        controller_mode_norm = str(controller_mode).strip().lower()
        controller_aliases = {
            "pi": "current",
            "gustafson": "gustafsson",
            "hairer": "hairer_lean",
            "ntss": "hairer_ntss",
        }
        controller_mode_norm = controller_aliases.get(controller_mode_norm, controller_mode_norm)
        if controller_mode_norm not in {"current", "current_legacy", "gustafsson", "hairer_lean", "hairer_ntss"}:
            raise ValueError(
                "theta_controller_mode must be one of: current, current_legacy, "
                "gustafsson, hairer_lean, hairer_ntss"
            )
        object.__setattr__(self, "controller_mode", controller_mode_norm)
        jacobian_reuse_mode_norm = str(jacobian_reuse_mode).strip().lower()
        jacobian_reuse_aliases = {
            "refresh": "refresh_each_iteration",
            "every_iteration": "refresh_each_iteration",
            "freeze": "freeze_attempt",
            "frozen": "freeze_attempt",
        }
        jacobian_reuse_mode_norm = jacobian_reuse_aliases.get(jacobian_reuse_mode_norm, jacobian_reuse_mode_norm)
        if jacobian_reuse_mode_norm not in {"refresh_each_iteration", "freeze_attempt"}:
            raise ValueError(
                "theta_jacobian_reuse_mode must be one of: "
                "refresh_each_iteration, freeze_attempt"
            )
        object.__setattr__(self, "jacobian_reuse_mode", jacobian_reuse_mode_norm)
        object.__setattr__(self, "jacobian_reuse_rtol", float(jacobian_reuse_rtol))
        object.__setattr__(self, "max_jacobian_age", int(max(0, max_jacobian_age)))
        lagged_reuse_mode_norm = str(lagged_response_reuse_mode).strip().lower()
        lagged_reuse_aliases = {
            "retry": "retry_only",
            "global": "global_state_drift",
            "drift": "global_state_drift",
        }
        lagged_reuse_mode_norm = lagged_reuse_aliases.get(lagged_reuse_mode_norm, lagged_reuse_mode_norm)
        if lagged_reuse_mode_norm not in {"retry_only", "global_state_drift"}:
            raise ValueError(
                "theta_lagged_response_reuse_mode must be one of: "
                "retry_only, global_state_drift"
            )
        object.__setattr__(self, "lagged_response_reuse_mode", lagged_reuse_mode_norm)
        object.__setattr__(self, "lagged_response_reuse_rtol", float(lagged_response_reuse_rtol))
        object.__setattr__(self, "lagged_response_reuse_atol", float(lagged_response_reuse_atol))
        object.__setattr__(self, "target_nonlinear_iterations", int(max(1, target_nonlinear_iterations)))
        object.__setattr__(self, "delta_reduction_factor", float(delta_reduction_factor))
        object.__setattr__(self, "tau_min", float(tau_min))


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class _ThetaStepState:
    t: Any
    y: Any
    dt: Any
    status: Any
    prev_error: Any = None
    prev_dt: Any = None
    recent_reject_count: Any = None
    regrowth_cooldown: Any = None
    easy_growth_streak: Any = None
    prev_theta_final: Any = None
    prev_newton_iter_count: Any = None
    reuse_state: Any = None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class _ThetaStepInfo:
    y: Any
    t: Any
    dt: Any
    accepted: Any
    failed: Any
    fail_code: Any
    converged: Any = None
    err_norm: Any = None
    diverged: Any = None
    nonfinite_stage_state: Any = None
    nonfinite_stage_residual: Any = None
    finite_f0: Any = None
    finite_z0: Any = None
    finite_initial_residual: Any = None
    newton_iter_count: Any = None
    final_residual_norm: Any = None
    final_delta_norm: Any = None
    theta_final: Any = None
    slow_contraction: Any = None
    residual_blowup: Any = None
    newton_nonfinite: Any = None
    lagged_reused: Any = None
    jacobian_reused: Any = None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class _ThetaAcceptedStepAttemptResult:
    trial_dt: Any
    trial_y: Any
    err_norm: Any
    converged: Any
    newton_iter_count: Any
    final_residual_norm: Any
    final_delta_norm: Any
    theta_final: Any
    slow_contraction: Any
    residual_blowup: Any
    newton_nonfinite: Any
    lagged_reused: Any
    jacobian_reused: Any
    lagged_response_cache_out: Any
    lagged_response_valid_out: Any
    lagged_reference_y_out: Any
    jacobian_out: Any
    cache_valid_out: Any
    cache_dt_out: Any
    cache_age_out: Any
    lu_out: Any
    piv_out: Any
    diverged_final: Any
    nonfinite_stage_state: Any
    nonfinite_stage_residual: Any
    finite_f0: Any
    finite_z0: Any
    finite_initial_residual: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class _ThetaAttemptContext:
    t: Any
    y: Any
    trial_dt: Any
    t_new: Any
    f_old: Any
    lagged_response: Any
    lagged_reference_y: Any
    lagged_response_reused: Any
    reuse_state: Any


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class _ThetaReuseState:
    lagged_response_cache: Any
    lagged_response_available: Any
    lagged_response_valid: Any
    lagged_reference_y: Any
    jacobian: Any
    cache_valid: Any
    cache_dt: Any
    cache_age: Any
    lu_factor: Any
    pivots: Any
    freeze_attempt_linearization: Any
    last_lagged_reused: Any
    last_jacobian_reused: Any
    last_linearization_dt: Any


def _theta_attempt_context_with_dt(
    attempt_context: _ThetaAttemptContext,
    trial_dt,
) -> _ThetaAttemptContext:
    return dataclasses.replace(
        attempt_context,
        trial_dt=trial_dt,
        t_new=attempt_context.t + trial_dt,
    )


def _theta_initial_reuse_state(state_dim, dtype) -> _ThetaReuseState:
    return _ThetaReuseState(
        lagged_response_cache=None,
        lagged_response_available=jnp.asarray(False),
        lagged_response_valid=jnp.asarray(False),
        lagged_reference_y=jnp.zeros((state_dim,), dtype=dtype),
        jacobian=jnp.zeros((state_dim, state_dim), dtype=dtype),
        cache_valid=jnp.asarray(False),
        cache_dt=jnp.asarray(0.0, dtype=dtype),
        cache_age=jnp.asarray(0, dtype=jnp.int32),
        lu_factor=jnp.eye(state_dim, dtype=dtype),
        pivots=jnp.arange(state_dim, dtype=jnp.int32),
        freeze_attempt_linearization=jnp.asarray(False),
        last_lagged_reused=jnp.asarray(False),
        last_jacobian_reused=jnp.asarray(False),
        last_linearization_dt=jnp.asarray(0.0, dtype=dtype),
    )


def _theta_make_reuse_state(
    *,
    lagged_response_cache,
    lagged_response_available,
    lagged_response_valid,
    lagged_reference_y,
    jacobian,
    cache_valid,
    cache_dt,
    cache_age,
    lu_factor,
    pivots,
    freeze_attempt_linearization,
    last_lagged_reused,
    last_jacobian_reused,
    last_linearization_dt,
):
    return _ThetaReuseState(
        lagged_response_cache=lagged_response_cache,
        lagged_response_available=lagged_response_available,
        lagged_response_valid=lagged_response_valid,
        lagged_reference_y=lagged_reference_y,
        jacobian=jacobian,
        cache_valid=cache_valid,
        cache_dt=cache_dt,
        cache_age=cache_age,
        lu_factor=lu_factor,
        pivots=pivots,
        freeze_attempt_linearization=freeze_attempt_linearization,
        last_lagged_reused=last_lagged_reused,
        last_jacobian_reused=last_jacobian_reused,
        last_linearization_dt=last_linearization_dt,
    )


def _theta_make_attempt_context(
    step_state: _ThetaStepState,
    *,
    t_final,
    flat_rhs,
    build_lagged_response,
    unpack_flat,
    project_flat,
    use_transport_lagged_response,
):
    trial_dt = jnp.minimum(step_state.dt, t_final - step_state.t)
    y_proj = _project_flat_state_if_needed(step_state.y, project_flat)
    lagged_response = (
        build_lagged_response(unpack_flat(y_proj))
        if (use_transport_lagged_response and build_lagged_response is not None)
        else None
    )
    return _ThetaAttemptContext(
        t=step_state.t,
        y=step_state.y,
        trial_dt=trial_dt,
        t_new=step_state.t + trial_dt,
        f_old=flat_rhs(step_state.t, step_state.y),
        lagged_response=lagged_response,
        lagged_reference_y=step_state.y,
        lagged_response_reused=jnp.asarray(False),
        reuse_state=step_state.reuse_state,
    )


def _theta_prepare_lagged_response(
    step_state: _ThetaStepState,
    *,
    use_transport_lagged_response,
    lagged_response_reuse_mode,
    lagged_response_reuse_rtol,
    lagged_response_reuse_atol,
    unpack_flat,
    project_flat,
    build_lagged_response,
):
    if not use_transport_lagged_response:
        return None, step_state.y, jnp.asarray(False)
    candidate_flat = _project_flat_state_if_needed(step_state.y, project_flat)
    candidate_state = unpack_flat(candidate_flat)
    reuse_state = step_state.reuse_state
    lagged_response_reused = jnp.asarray(False)
    can_reuse = reuse_state.lagged_response_valid
    if lagged_response_reuse_mode == "global_state_drift":
        lagged_reuse_metric = _lagged_response_global_reuse_metric(
            candidate_flat,
            reuse_state.lagged_reference_y,
            atol=lagged_response_reuse_atol,
            rtol=lagged_response_reuse_rtol,
        )
        can_reuse = jnp.logical_and(can_reuse, lagged_reuse_metric <= jnp.asarray(1.0, dtype=candidate_flat.dtype))
    if lagged_response_reuse_mode == "retry_only":
        can_reuse = jnp.logical_and(can_reuse, step_state.recent_reject_count > jnp.asarray(0, dtype=jnp.int32))
    lagged_response_reused = jnp.asarray(can_reuse)

    def _reuse_cached(_):
        return reuse_state.lagged_response_cache

    def _rebuild_cached(_):
        if build_lagged_response is None:
            return None
        return build_lagged_response(candidate_state)

    lagged_response = jax.lax.cond(
        can_reuse,
        _reuse_cached,
        _rebuild_cached,
        operand=None,
    )
    lagged_reference_y = jax.lax.cond(
        can_reuse,
        lambda _: reuse_state.lagged_reference_y,
        lambda _: candidate_flat,
        operand=None,
    )
    return lagged_response, lagged_reference_y, lagged_response_reused


def _theta_step_info_from_attempt(
    step_state: _ThetaStepState,
    *,
    attempt_result: _ThetaAcceptedStepAttemptResult,
    next_y,
    next_t,
    accepted,
    fail_now,
    fail_code,
    dtype,
) -> _ThetaStepInfo:
    return _ThetaStepInfo(
        y=jnp.where(accepted, next_y, step_state.y),
        t=jnp.where(accepted, next_t, step_state.t),
        dt=jnp.where(accepted, attempt_result.trial_dt, jnp.asarray(0.0, dtype=dtype)),
        accepted=accepted,
        failed=fail_now,
        fail_code=fail_code,
        converged=attempt_result.converged,
        err_norm=attempt_result.err_norm,
        diverged=attempt_result.diverged_final,
        nonfinite_stage_state=attempt_result.nonfinite_stage_state,
        nonfinite_stage_residual=attempt_result.nonfinite_stage_residual,
        finite_f0=attempt_result.finite_f0,
        finite_z0=attempt_result.finite_z0,
        finite_initial_residual=attempt_result.finite_initial_residual,
        newton_iter_count=attempt_result.newton_iter_count,
        final_residual_norm=attempt_result.final_residual_norm,
        final_delta_norm=attempt_result.final_delta_norm,
        theta_final=attempt_result.theta_final,
        slow_contraction=attempt_result.slow_contraction,
        residual_blowup=attempt_result.residual_blowup,
        newton_nonfinite=attempt_result.newton_nonfinite,
        lagged_reused=attempt_result.lagged_reused,
        jacobian_reused=attempt_result.jacobian_reused,
    )


def _theta_step_transition_from_attempt(
    step_state: _ThetaStepState,
    *,
    attempt_result: _ThetaAcceptedStepAttemptResult,
    n_accepted,
    next_dt_if_accepted,
    next_dt_if_rejected,
    next_prev_error_if_accepted,
    next_prev_dt_if_accepted,
    next_recent_reject_count_if_accepted,
    next_recent_reject_count_if_rejected,
    next_regrowth_cooldown_if_accepted,
    next_regrowth_cooldown_if_rejected,
    next_easy_growth_streak_if_accepted,
    next_easy_growth_streak_if_rejected,
    next_reuse_state_if_accepted,
    next_reuse_state_if_rejected,
    project_flat,
    dtype,
):
    accepted = attempt_result.converged
    t_new = step_state.t + attempt_result.trial_dt
    next_y = _project_flat_state_if_needed(attempt_result.trial_y, project_flat)
    fail_now = jnp.logical_not(accepted)
    fail_code = jnp.where(accepted, jnp.asarray(0, dtype=jnp.int32), jnp.asarray(1, dtype=jnp.int32))
    status_next = jnp.asarray(
        [fail_now.astype(jnp.int32), fail_code, n_accepted + jnp.where(accepted, 1, 0)],
        dtype=jnp.int32,
    )
    next_state = _ThetaStepState(
        t=jnp.where(accepted, t_new, step_state.t),
        y=jnp.where(accepted, next_y, step_state.y),
        dt=jnp.where(accepted, next_dt_if_accepted, next_dt_if_rejected),
        status=status_next,
        prev_error=jnp.where(accepted, next_prev_error_if_accepted, step_state.prev_error),
        prev_dt=jnp.where(accepted, next_prev_dt_if_accepted, step_state.prev_dt),
        recent_reject_count=jnp.where(accepted, next_recent_reject_count_if_accepted, next_recent_reject_count_if_rejected),
        regrowth_cooldown=jnp.where(accepted, next_regrowth_cooldown_if_accepted, next_regrowth_cooldown_if_rejected),
        easy_growth_streak=jnp.where(accepted, next_easy_growth_streak_if_accepted, next_easy_growth_streak_if_rejected),
        prev_theta_final=attempt_result.theta_final,
        prev_newton_iter_count=attempt_result.newton_iter_count,
        reuse_state=jax.lax.cond(
            accepted,
            lambda _: next_reuse_state_if_accepted,
            lambda _: next_reuse_state_if_rejected,
            operand=None,
        ),
    )
    step_info = _theta_step_info_from_attempt(
        step_state,
        attempt_result=attempt_result,
        next_y=next_y,
        next_t=t_new,
        accepted=accepted,
        fail_now=fail_now,
        fail_code=fail_code,
        dtype=dtype,
    )
    return next_state, step_info


def _theta_controller_update(
    *,
    step_state: _ThetaStepState,
    trial_dt,
    converged,
    residual_norm,
    newton_iter_count,
    theta_final,
    slow_contraction,
    lagged_reused,
    jacobian_reused,
    dt_min,
    dt_max,
    safety_factor,
    min_step_factor,
    max_step_factor,
    controller_mode,
    dtype,
):
    """Theta controller modeled after the well-behaved Radau controller shell."""
    accepted = jnp.asarray(converged)
    safe_error = jnp.maximum(residual_norm, jnp.asarray(1.0e-12, dtype=dtype))
    prev_error = jnp.maximum(step_state.prev_error, jnp.asarray(1.0e-12, dtype=dtype))
    controller_alpha = jnp.asarray(0.35, dtype=dtype)
    controller_beta = jnp.asarray(0.3, dtype=dtype) * controller_alpha
    controller_gamma = jnp.asarray(0.1, dtype=dtype) * controller_alpha
    prev_dt_safe = jnp.maximum(step_state.prev_dt, dt_min)
    prev_dt_available = step_state.prev_dt > jnp.asarray(0.0, dtype=dtype)
    step_ratio_prev = jnp.where(prev_dt_available, trial_dt / prev_dt_safe, jnp.asarray(1.0, dtype=dtype))
    gustafsson_damping = jnp.where(
        prev_dt_available,
        jnp.clip(
            jnp.where(
                step_ratio_prev > jnp.asarray(1.0, dtype=dtype),
                step_ratio_prev ** (-controller_gamma),
                jnp.asarray(1.0, dtype=dtype),
            ),
            jnp.asarray(0.85, dtype=dtype),
            jnp.asarray(1.0, dtype=dtype),
        ),
        jnp.asarray(1.0, dtype=dtype),
    )
    growth_current = safety_factor * safe_error ** (-controller_alpha) * prev_error ** controller_beta * gustafsson_damping
    growth_current = jnp.clip(growth_current, min_step_factor, max_step_factor)
    growth_pi = safety_factor * safe_error ** (-controller_alpha) * prev_error ** controller_beta
    growth_pi = jnp.clip(growth_pi, min_step_factor, max_step_factor)
    growth_predictive = jnp.where(
        prev_dt_available,
        safety_factor
        * (prev_dt_safe / jnp.maximum(trial_dt, dt_min))
        * (prev_error / jnp.maximum(safe_error * safe_error, jnp.asarray(1.0e-12, dtype=dtype))) ** controller_alpha,
        growth_pi,
    )
    growth_predictive = jnp.clip(growth_predictive, min_step_factor, max_step_factor)
    nit_ref = jnp.asarray(20.0, dtype=dtype)
    newton_iter_float = jnp.asarray(newton_iter_count, dtype=dtype)
    fac_ntss = jnp.minimum(
        safety_factor,
        ((jnp.asarray(2.0, dtype=dtype) * nit_ref + jnp.asarray(1.0, dtype=dtype)) * safety_factor)
        / jnp.maximum(newton_iter_float + jnp.asarray(2.0, dtype=dtype) * nit_ref, jnp.asarray(1.0, dtype=dtype)),
    )
    growth_ntss = jnp.asarray(1.0, dtype=dtype) / jnp.clip(
        safe_error ** controller_alpha / jnp.maximum(fac_ntss, jnp.asarray(1.0e-12, dtype=dtype)),
        jnp.asarray(1.0, dtype=dtype) / jnp.asarray(8.0, dtype=dtype),
        jnp.asarray(1.0, dtype=dtype) / jnp.asarray(0.75, dtype=dtype),
    )
    growth_current_legacy = safety_factor * safe_error ** (-controller_alpha) * prev_error ** controller_beta
    growth_current_legacy = jnp.clip(growth_current_legacy, min_step_factor, max_step_factor)
    use_gustafsson_controller = controller_mode == "gustafsson"
    use_current_legacy_controller = controller_mode == "current_legacy"
    use_hairer_lean_controller = controller_mode == "hairer_lean"
    use_hairer_ntss_controller = controller_mode == "hairer_ntss"
    growth = jnp.where(
        use_gustafsson_controller,
        jnp.minimum(growth_pi, growth_predictive),
        jnp.where(
            use_hairer_ntss_controller,
            growth_ntss,
            jnp.where(
                use_hairer_lean_controller,
                jnp.clip(safety_factor * safe_error ** (-controller_alpha), min_step_factor, max_step_factor),
                jnp.where(use_current_legacy_controller, growth_current_legacy, growth_current),
            ),
        ),
    )
    difficult_theta = jnp.asarray(0.03, dtype=dtype)
    very_difficult_theta = jnp.asarray(0.10, dtype=dtype)
    easy_theta = jnp.asarray(0.01, dtype=dtype)
    easy_error = jnp.asarray(0.05, dtype=dtype)
    recovery_theta = jnp.asarray(0.02, dtype=dtype)
    recovery_error = jnp.asarray(0.1, dtype=dtype)
    reused_any = jnp.logical_or(lagged_reused, jacobian_reused)
    reused_both = jnp.logical_and(lagged_reused, jacobian_reused)
    difficult_accept = jnp.logical_or(
        slow_contraction,
        jnp.logical_or(theta_final >= difficult_theta, newton_iter_count >= jnp.asarray(6, dtype=jnp.int32)),
    )
    very_difficult_accept = jnp.logical_or(theta_final >= very_difficult_theta, newton_iter_count >= jnp.asarray(8, dtype=jnp.int32))
    easy_accept = jnp.logical_and(
        jnp.logical_and(safe_error <= easy_error, theta_final <= easy_theta),
        jnp.logical_and(newton_iter_count <= jnp.asarray(4, dtype=jnp.int32), jnp.logical_not(slow_contraction)),
    )
    recovery_ready = jnp.logical_and(
        jnp.logical_and(safe_error <= recovery_error, theta_final <= recovery_theta),
        jnp.logical_and(newton_iter_count <= jnp.asarray(5, dtype=jnp.int32), jnp.logical_not(slow_contraction)),
    )
    difficulty_growth_cap = jnp.where(
        very_difficult_accept,
        jnp.asarray(1.0, dtype=dtype),
        jnp.where(difficult_accept, jnp.asarray(1.25, dtype=dtype), jnp.asarray(1.5, dtype=dtype)),
    )
    reuse_growth_cap = jnp.where(
        reused_both,
        jnp.where(difficult_accept, jnp.asarray(1.0, dtype=dtype), jnp.asarray(1.15, dtype=dtype)),
        jnp.where(
            reused_any,
            jnp.where(difficult_accept, jnp.asarray(1.0, dtype=dtype), jnp.asarray(1.25, dtype=dtype)),
            max_step_factor,
        ),
    )
    post_reject_growth_cap = jnp.where(
        jnp.logical_or(use_hairer_lean_controller, use_hairer_ntss_controller),
        max_step_factor,
        jnp.where(
            use_gustafsson_controller,
            jnp.where(step_state.recent_reject_count > 0, jnp.where(recovery_ready, jnp.asarray(1.45, dtype=dtype), jnp.asarray(1.15, dtype=dtype)), max_step_factor),
            jnp.where(step_state.regrowth_cooldown > 0, jnp.where(recovery_ready, jnp.asarray(1.35, dtype=dtype), jnp.asarray(1.15, dtype=dtype)), max_step_factor),
        ),
    )
    streak_growth_cap = jnp.where(
        jnp.logical_or(use_gustafsson_controller, use_hairer_lean_controller, use_hairer_ntss_controller),
        max_step_factor,
        jnp.where(step_state.easy_growth_streak >= jnp.asarray(1, dtype=jnp.int32), max_step_factor, jnp.asarray(1.75, dtype=dtype)),
    )
    growth_cap = jnp.where(
        jnp.logical_or(use_hairer_lean_controller, use_hairer_ntss_controller),
        max_step_factor,
        jnp.minimum(
            jnp.minimum(jnp.minimum(max_step_factor, difficulty_growth_cap), reuse_growth_cap),
            jnp.minimum(post_reject_growth_cap, streak_growth_cap),
        ),
    )
    growth = jnp.clip(growth, min_step_factor, growth_cap)
    growth = jnp.where(
        jnp.logical_and(
            jnp.logical_not(jnp.logical_or(use_hairer_lean_controller, use_hairer_ntss_controller)),
            jnp.logical_and(difficult_accept, growth <= jnp.asarray(1.35, dtype=dtype)),
        ),
        jnp.asarray(1.0, dtype=dtype),
        growth,
    )
    growth = jnp.where(
        jnp.logical_and(
            jnp.logical_not(jnp.logical_or(use_hairer_lean_controller, use_hairer_ntss_controller)),
            jnp.logical_and(step_state.regrowth_cooldown > 0, recovery_ready),
        ),
        jnp.maximum(growth, jnp.asarray(1.2, dtype=dtype)),
        growth,
    )
    next_dt_accept = jnp.clip(trial_dt * growth, dt_min, dt_max)

    retry_count_next = jnp.minimum(step_state.recent_reject_count + jnp.asarray(1, dtype=jnp.int32), jnp.asarray(3, dtype=jnp.int32))
    retry_shrink = jnp.where(
        retry_count_next >= jnp.asarray(3, dtype=jnp.int32),
        jnp.asarray(0.6, dtype=dtype),
        jnp.where(
            retry_count_next >= jnp.asarray(2, dtype=jnp.int32),
            jnp.asarray(0.8, dtype=dtype),
            jnp.asarray(1.0, dtype=dtype),
        ),
    )
    retry_shrink = jnp.where(
        reused_both,
        retry_shrink * jnp.asarray(0.8, dtype=dtype),
        jnp.where(reused_any, retry_shrink * jnp.asarray(0.9, dtype=dtype), retry_shrink),
    )
    reduced_dt = jnp.maximum(jnp.minimum(trial_dt * retry_shrink, trial_dt * jnp.asarray(0.5, dtype=dtype)), dt_min)
    regrowth_cooldown_next = jnp.where(
        jnp.logical_or(use_gustafsson_controller, use_hairer_ntss_controller),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.maximum(
            step_state.regrowth_cooldown - jnp.where(easy_accept, jnp.asarray(2, dtype=jnp.int32), jnp.asarray(1, dtype=jnp.int32)),
            jnp.asarray(0, dtype=jnp.int32),
        ),
    )
    reject_regrowth_cooldown_next = jnp.where(
        jnp.logical_or(use_gustafsson_controller, use_hairer_ntss_controller),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.where(
            reused_both,
            jnp.asarray(3, dtype=jnp.int32),
            jnp.where(reused_any, jnp.asarray(2, dtype=jnp.int32), jnp.asarray(1, dtype=jnp.int32)),
        ),
    )
    easy_growth_streak_next = jnp.where(
        jnp.logical_or(use_gustafsson_controller, use_hairer_lean_controller, use_hairer_ntss_controller),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.where(
            jnp.logical_and(easy_accept, jnp.logical_not(reused_any)),
            jnp.minimum(step_state.easy_growth_streak + jnp.asarray(1, dtype=jnp.int32), jnp.asarray(3, dtype=jnp.int32)),
            jnp.asarray(0, dtype=jnp.int32),
        ),
    )
    return {
        "accepted": accepted,
        "growth": growth,
        "next_dt_accept": next_dt_accept,
        "next_dt_reject": reduced_dt,
        "safe_error": safe_error,
        "easy_accept": easy_accept,
        "regrowth_cooldown_next": regrowth_cooldown_next,
        "reject_regrowth_cooldown_next": reject_regrowth_cooldown_next,
        "easy_growth_streak_next": easy_growth_streak_next,
        "retry_count_next": retry_count_next,
    }


class ThetaMethodSolver(_ThetaSolverConfig):
    def solve(self, state, vector_field: Callable, *args, **kwargs):
        STATUS_FAILED = 0
        STATUS_FAIL_CODE = 1
        STATUS_N_ACCEPTED = 2
        species = _extract_species_from_args(args)
        temperature_active_mask, fixed_temperature_profile = _extract_fixed_temperature_projection(vector_field)
        density_floor, temperature_floor = _extract_state_regularization(vector_field)
        state = _project_state_to_quasi_neutrality(
            state,
            species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
            density_floor=density_floor,
            temperature_floor=temperature_floor,
        )
        flat_state0, unpack_flat, _unpack_packed, _pack_state, project_flat = _make_solver_state_transform(
            state,
            species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
            density_floor=density_floor,
            temperature_floor=temperature_floor,
        )
        dtype = flat_state0.dtype
        theta = jnp.asarray(self.theta_implicit, dtype=dtype)
        t0 = jnp.asarray(self.t0, dtype=dtype)
        t_final = jnp.asarray(self.t1, dtype=dtype)
        base_dt = jnp.asarray(self.dt, dtype=dtype)
        dt_min = jnp.asarray(self.min_step, dtype=dtype)
        max_total_steps = int(max(1, self.max_steps))
        state_dim = flat_state0.shape[0]
        identity_n = jnp.eye(state_dim, dtype=dtype)
        flat_rhs = _flat_rhs_factory(unpack_flat, vector_field, args, kwargs, project_flat=project_flat)
        build_lagged_response, _ = _lagged_response_hooks(vector_field)
        flat_rhs_with_lagged_response = _flat_rhs_with_lagged_response_factory(
            unravel=unpack_flat,
            vector_field=vector_field,
            args=args,
            kwargs=kwargs,
            project_flat=project_flat,
        )
        predictor_mode = getattr(self, "predictor_mode", "linearized")
        rhs_mode = str(getattr(self, "rhs_mode", "black_box")).strip().lower()
        use_lagged_linear_response = rhs_mode == "lagged_linear_state"
        use_transport_lagged_response = rhs_mode in {"lagged_transport_response", "lagged_response"}
        if use_transport_lagged_response and build_lagged_response is None:
            raise ValueError(
                "Theta lagged transport response mode requires a vector field with "
                "build_lagged_response(...) and evaluate_with_lagged_response(...)."
            )
        n_linearized_solves = 1 + (self.n_corrector_steps if self.use_predictor_corrector else 0)

        def _single_theta_step(attempt_context: _ThetaAttemptContext):
            flat_y = attempt_context.y
            t_value = attempt_context.t
            h_value = attempt_context.trial_dt
            f_old = attempt_context.f_old
            t_new = attempt_context.t_new
            lagged_response = attempt_context.lagged_response
            guess0 = _project_flat_state_if_needed(flat_y + h_value * f_old, project_flat)
            f_ref_new = flat_rhs(t_new, flat_y)
            jacobian_ref = jax.jacfwd(lambda y: flat_rhs(t_new, y))(flat_y)

            def _eval_new_rhs(y_value):
                y_proj = _project_flat_state_if_needed(y_value, project_flat)
                if lagged_response is not None:
                    return flat_rhs_with_lagged_response(t_new, y_proj, lagged_response)
                if use_lagged_linear_response:
                    return f_ref_new + jacobian_ref @ (y_proj - flat_y)
                return flat_rhs(t_new, y_proj)

            if predictor_mode == "euler":
                y_new = guess0
                linear_ok = jnp.asarray(True)
                f_new = _eval_new_rhs(y_new)
                residual = y_new - flat_y - h_value * (
                    (jnp.asarray(1.0, dtype=dtype) - theta) * f_old + theta * f_new
                )
                residual_norm = jnp.linalg.norm(residual)
                converged = residual_norm <= self.tol
                return _ThetaAcceptedStepAttemptResult(
                    trial_dt=h_value,
                    trial_y=y_new,
                    err_norm=residual_norm,
                    converged=converged,
                    newton_iter_count=jnp.asarray(1, dtype=jnp.int32),
                    final_residual_norm=residual_norm,
                    final_delta_norm=jnp.linalg.norm(y_new - flat_y),
                    theta_final=jnp.asarray(0.0, dtype=dtype),
                    slow_contraction=jnp.asarray(False),
                    residual_blowup=jnp.asarray(False),
                    newton_nonfinite=jnp.logical_or(
                        jnp.logical_not(jnp.all(jnp.isfinite(y_new))),
                        jnp.logical_not(jnp.isfinite(residual_norm)),
                    ),
                    lagged_reused=jnp.asarray(lagged_response is not None),
                    jacobian_reused=jnp.asarray(False),
                    lagged_response_cache_out=lagged_response,
                    lagged_response_valid_out=jnp.asarray(False),
                    lagged_reference_y_out=flat_y,
                    jacobian_out=jnp.zeros((state_dim, state_dim), dtype=dtype),
                    cache_valid_out=jnp.asarray(False),
                    cache_dt_out=h_value,
                    cache_age_out=jnp.asarray(0, dtype=jnp.int32),
                    lu_out=identity_n,
                    piv_out=jnp.arange(state_dim, dtype=jnp.int32),
                    diverged_final=jnp.logical_not(converged),
                    nonfinite_stage_state=jnp.logical_not(jnp.all(jnp.isfinite(y_new))),
                    nonfinite_stage_residual=jnp.logical_not(jnp.isfinite(residual_norm)),
                    finite_f0=jnp.asarray(True),
                    finite_z0=jnp.all(jnp.isfinite(flat_y)),
                    finite_initial_residual=jnp.asarray(True),
                )

            jacobian_guess0 = jnp.where(
                use_lagged_linear_response,
                jacobian_ref,
                jax.jacfwd(lambda y: flat_rhs(t_new, y))(guess0),
            )
            system0 = identity_n - h_value * theta * jacobian_guess0
            lu0, piv0 = jax.scipy.linalg.lu_factor(system0)
            linearization_finite = jnp.logical_and(jnp.all(jnp.isfinite(system0)), jnp.all(jnp.isfinite(lu0)))

            def _corrector_body(_, carry):
                guess, linear_ok, prev_delta_norm, theta_est, final_delta_norm = carry
                guess_proj = _project_flat_state_if_needed(guess, project_flat)
                f_guess = _eval_new_rhs(guess_proj)
                affine_rhs = flat_y + h_value * (
                    (jnp.asarray(1.0, dtype=dtype) - theta) * f_old
                    + theta * (f_guess - jacobian_guess0 @ guess_proj)
                )
                next_guess = jax.scipy.linalg.lu_solve((lu0, piv0), affine_rhs)
                finite = jnp.logical_and(jnp.all(jnp.isfinite(next_guess)), linearization_finite)
                next_guess = jnp.where(finite, next_guess, guess_proj)
                next_guess = _project_flat_state_if_needed(next_guess, project_flat)
                delta_norm = jnp.linalg.norm(next_guess - guess_proj)
                theta_candidate = jnp.where(
                    prev_delta_norm > jnp.asarray(0.0, dtype=dtype),
                    delta_norm / jnp.maximum(prev_delta_norm, jnp.asarray(1.0e-30, dtype=dtype)),
                    theta_est,
                )
                theta_next = jnp.where(
                    prev_delta_norm > jnp.asarray(0.0, dtype=dtype),
                    theta_candidate,
                    theta_est,
                )
                return (
                    next_guess,
                    jnp.logical_and(linear_ok, finite),
                    delta_norm,
                    theta_next,
                    delta_norm,
                )

            y_new, linear_ok, _prev_delta_norm, theta_final, final_delta_norm = jax.lax.fori_loop(
                0,
                n_linearized_solves,
                _corrector_body,
                (
                    guess0,
                    jnp.asarray(True),
                    jnp.asarray(0.0, dtype=dtype),
                    jnp.asarray(0.0, dtype=dtype),
                    jnp.asarray(0.0, dtype=dtype),
                ),
            )
            f_new = _eval_new_rhs(y_new)
            residual = y_new - flat_y - h_value * (
                (jnp.asarray(1.0, dtype=dtype) - theta) * f_old + theta * f_new
            )
            residual_norm = jnp.linalg.norm(residual)
            converged = jnp.logical_and(linear_ok, residual_norm <= self.tol)
            return _ThetaAcceptedStepAttemptResult(
                trial_dt=h_value,
                trial_y=y_new,
                err_norm=residual_norm,
                converged=converged,
                newton_iter_count=jnp.asarray(n_linearized_solves, dtype=jnp.int32),
                final_residual_norm=residual_norm,
                final_delta_norm=final_delta_norm,
                theta_final=theta_final,
                slow_contraction=theta_final >= jnp.asarray(0.8, dtype=dtype),
                residual_blowup=jnp.asarray(False),
                newton_nonfinite=jnp.logical_or(
                    jnp.logical_not(jnp.all(jnp.isfinite(y_new))),
                    jnp.logical_not(jnp.isfinite(residual_norm)),
                ),
                lagged_reused=jnp.asarray(lagged_response is not None),
                jacobian_reused=jnp.asarray(predictor_mode != "euler"),
                lagged_response_cache_out=lagged_response,
                lagged_response_valid_out=jnp.asarray(False),
                lagged_reference_y_out=flat_y,
                jacobian_out=jacobian_guess0,
                cache_valid_out=jnp.asarray(True),
                cache_dt_out=h_value,
                cache_age_out=jnp.asarray(0, dtype=jnp.int32),
                lu_out=lu0,
                piv_out=piv0,
                diverged_final=jnp.logical_not(converged),
                nonfinite_stage_state=jnp.logical_not(jnp.all(jnp.isfinite(y_new))),
                nonfinite_stage_residual=jnp.logical_not(jnp.isfinite(residual_norm)),
                finite_f0=jnp.asarray(True),
                finite_z0=jnp.all(jnp.isfinite(flat_y)),
                finite_initial_residual=jnp.asarray(True),
            )

        def step_fn(step_state: _ThetaStepState, _):
            failed = step_state.status[STATUS_FAILED] != 0
            fail_code = step_state.status[STATUS_FAIL_CODE]
            n_accepted = step_state.status[STATUS_N_ACCEPTED]

            def _skip(_):
                return step_state, _ThetaStepInfo(
                    y=step_state.y,
                    t=step_state.t,
                    dt=jnp.asarray(0.0, dtype=dtype),
                    accepted=jnp.asarray(False),
                    failed=failed,
                    fail_code=fail_code,
                )

            def _run(_):
                attempt_context = _theta_make_attempt_context(
                    step_state,
                    t_final=t_final,
                    flat_rhs=flat_rhs,
                    build_lagged_response=build_lagged_response,
                    unpack_flat=unpack_flat,
                    project_flat=project_flat,
                    use_transport_lagged_response=use_transport_lagged_response,
                )
                attempt_result = _single_theta_step(attempt_context)
                attempt_reuse_state = _theta_make_reuse_state(
                    lagged_response_cache=attempt_result.lagged_response_cache_out,
                    lagged_response_available=jnp.asarray(attempt_context.lagged_response is not None),
                    lagged_response_valid=attempt_result.lagged_response_valid_out,
                    lagged_reference_y=attempt_result.lagged_reference_y_out,
                    jacobian=attempt_result.jacobian_out,
                    cache_valid=attempt_result.cache_valid_out,
                    cache_dt=attempt_result.cache_dt_out,
                    cache_age=attempt_result.cache_age_out,
                    lu_factor=attempt_result.lu_out,
                    pivots=attempt_result.piv_out,
                    freeze_attempt_linearization=jnp.asarray(False),
                    last_lagged_reused=attempt_result.lagged_reused,
                    last_jacobian_reused=attempt_result.jacobian_reused,
                    last_linearization_dt=attempt_result.trial_dt,
                )
                return _theta_step_transition_from_attempt(
                    step_state,
                    attempt_result=attempt_result,
                    n_accepted=n_accepted,
                    next_dt_if_accepted=step_state.dt,
                    next_dt_if_rejected=step_state.dt,
                    next_prev_error_if_accepted=jnp.maximum(attempt_result.err_norm, jnp.asarray(1.0e-12, dtype=dtype)),
                    next_prev_dt_if_accepted=attempt_result.trial_dt,
                    next_recent_reject_count_if_accepted=jnp.asarray(0, dtype=jnp.int32),
                    next_recent_reject_count_if_rejected=step_state.recent_reject_count,
                    next_regrowth_cooldown_if_accepted=step_state.regrowth_cooldown,
                    next_regrowth_cooldown_if_rejected=step_state.regrowth_cooldown,
                    next_easy_growth_streak_if_accepted=step_state.easy_growth_streak,
                    next_easy_growth_streak_if_rejected=step_state.easy_growth_streak,
                    next_reuse_state_if_accepted=attempt_reuse_state,
                    next_reuse_state_if_rejected=attempt_reuse_state,
                    project_flat=project_flat,
                    dtype=dtype,
                )

            return jax.lax.cond(failed, _skip, _run, operand=None)

        step_state0 = _ThetaStepState(
            t=t0,
            y=flat_state0,
            dt=base_dt,
            status=jnp.asarray([0, 0, 0], dtype=jnp.int32),
            prev_error=jnp.asarray(1.0, dtype=dtype),
            prev_dt=jnp.asarray(0.0, dtype=dtype),
            recent_reject_count=jnp.asarray(0, dtype=jnp.int32),
            regrowth_cooldown=jnp.asarray(0, dtype=jnp.int32),
            easy_growth_streak=jnp.asarray(0, dtype=jnp.int32),
            prev_theta_final=jnp.asarray(0.0, dtype=dtype),
            prev_newton_iter_count=jnp.asarray(0, dtype=jnp.int32),
            reuse_state=_theta_initial_reuse_state(state_dim, dtype),
        )
        save_n = getattr(self, "save_n", None)
        save_n = max(1, int(save_n)) if save_n is not None else 1
        stop_after_accepted_steps = getattr(self, "stop_after_accepted_steps", None)
        (
            step_state_f,
            _,
            _,
            ys_saved,
            ts_saved,
            dts_saved,
            accepted_mask_saved,
            failed_mask_saved,
            fail_codes_saved,
            last_attempt_accepted,
            last_attempt_converged,
            last_attempt_err_norm,
            last_attempt_fail_code,
            last_attempt_diverged,
            last_attempt_nonfinite_stage_state,
            last_attempt_nonfinite_stage_residual,
            last_attempt_finite_f0,
            last_attempt_finite_z0,
            last_attempt_finite_initial_residual,
            last_attempt_newton_iter_count,
            last_attempt_final_residual_norm,
            last_attempt_final_delta_norm,
            last_attempt_theta_final,
            last_attempt_slow_contraction,
            last_attempt_residual_blowup,
            last_attempt_newton_nonfinite,
            last_attempt_lagged_reused,
            last_attempt_jacobian_reused,
        ) = _run_saved_loop(
            step_state0=step_state0,
            step_fn=step_fn,
            save_n=save_n,
            t0=t0,
            t_final=t_final,
            state_dim=state_dim,
            dtype=dtype,
            max_total_steps=max_total_steps,
            stop_after_accepted_steps=stop_after_accepted_steps,
        )
        failed_f = step_state_f.status[STATUS_FAILED] != 0
        fail_code_f = step_state_f.status[STATUS_FAIL_CODE]
        n_acc_f = step_state_f.status[STATUS_N_ACCEPTED]
        accepted_limit_hit = _accepted_step_limit_reached(step_state_f, stop_after_accepted_steps)
        return _finalize_custom_solver_output(
            ys_saved,
            ts_saved,
            dts_saved,
            accepted_mask_saved,
            failed_mask_saved,
            fail_codes_saved,
            step_state_f.y,
            step_state_f.t,
            jnp.logical_or(step_state_f.t >= (t_final - 1.0e-15), accepted_limit_hit),
            failed_f,
            fail_code_f,
            n_acc_f,
            last_attempt_accepted,
            last_attempt_converged,
            last_attempt_err_norm,
            last_attempt_fail_code,
            last_attempt_diverged,
            last_attempt_nonfinite_stage_state,
            last_attempt_nonfinite_stage_residual,
            last_attempt_finite_f0,
            last_attempt_finite_z0,
            last_attempt_finite_initial_residual,
            last_attempt_newton_iter_count,
            last_attempt_final_residual_norm,
            last_attempt_final_delta_norm,
            last_attempt_theta_final,
            last_attempt_slow_contraction,
            last_attempt_residual_blowup,
            last_attempt_newton_nonfinite,
            last_attempt_lagged_reused,
            last_attempt_jacobian_reused,
            step_state_f.reuse_state,
            unpack_flat,
            state,
            species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
            density_floor=density_floor,
            temperature_floor=temperature_floor,
        )


class NewtonThetaMethodSolver(_ThetaNewtonSolverConfig):
    def solve(self, state, vector_field: Callable, *args, **kwargs):
        STATUS_FAILED = 0
        STATUS_FAIL_CODE = 1
        STATUS_N_ACCEPTED = 2
        species = _extract_species_from_args(args)
        temperature_active_mask, fixed_temperature_profile = _extract_fixed_temperature_projection(vector_field)
        density_floor, temperature_floor = _extract_state_regularization(vector_field)
        state = _project_state_to_quasi_neutrality(
            state,
            species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
            density_floor=density_floor,
            temperature_floor=temperature_floor,
        )
        flat_state0, unpack_flat, _unpack_packed, _pack_state, project_flat = _make_solver_state_transform(
            state,
            species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
            density_floor=density_floor,
            temperature_floor=temperature_floor,
        )
        dtype = flat_state0.dtype
        theta = jnp.asarray(self.theta_implicit, dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)
        t0 = jnp.asarray(self.t0, dtype=dtype)
        t_final = jnp.asarray(self.t1, dtype=dtype)
        dt_min = jnp.asarray(self.min_step, dtype=dtype)
        dt_max = jnp.asarray(self.max_step, dtype=dtype)
        base_dt = jnp.clip(jnp.asarray(self.dt, dtype=dtype), dt_min, dt_max)
        max_total_steps = int(max(1, self.max_steps))
        state_dim = flat_state0.shape[0]
        identity_n = jnp.eye(state_dim, dtype=dtype)
        flat_rhs = _flat_rhs_factory(unpack_flat, vector_field, args, kwargs, project_flat=project_flat)
        build_lagged_response, _ = _lagged_response_hooks(vector_field)
        flat_rhs_with_lagged_response = _flat_rhs_with_lagged_response_factory(
            unravel=unpack_flat,
            vector_field=vector_field,
            args=args,
            kwargs=kwargs,
            project_flat=project_flat,
        )
        predictor_mode = getattr(self, "predictor_mode", "linearized")
        controller_mode = str(getattr(self, "controller_mode", "current")).strip().lower()
        rhs_mode = str(getattr(self, "rhs_mode", "black_box")).strip().lower()
        lagged_response_reuse_mode = str(getattr(self, "lagged_response_reuse_mode", "retry_only")).strip().lower()
        lagged_response_reuse_rtol = jnp.asarray(getattr(self, "lagged_response_reuse_rtol", 5.0e-2), dtype=dtype)
        lagged_response_reuse_atol = jnp.asarray(getattr(self, "lagged_response_reuse_atol", 1.0e-8), dtype=dtype)
        use_lagged_linear_response = rhs_mode == "lagged_linear_state"
        use_transport_lagged_response = rhs_mode in {"lagged_transport_response", "lagged_response"}
        if use_transport_lagged_response and build_lagged_response is None:
            raise ValueError(
                "Theta lagged transport response mode requires a vector field with "
                "build_lagged_response(...) and evaluate_with_lagged_response(...)."
            )
        n_linearized_solves = 1 + (self.n_corrector_steps if self.use_predictor_corrector else 0)
        safety_factor = jnp.asarray(self.safety_factor, dtype=dtype)
        min_step_factor = jnp.asarray(self.min_step_factor, dtype=dtype)
        max_step_factor = jnp.asarray(self.max_step_factor, dtype=dtype)
        target_nonlinear_iterations = jnp.asarray(self.target_nonlinear_iterations, dtype=dtype)
        delta_reduction_factor = jnp.asarray(self.delta_reduction_factor, dtype=dtype)
        tau_min = jnp.asarray(self.tau_min, dtype=dtype)
        jacobian_reuse_rtol = jnp.asarray(getattr(self, "jacobian_reuse_rtol", 0.1), dtype=dtype)
        max_jacobian_age = jnp.asarray(getattr(self, "max_jacobian_age", 8), dtype=jnp.int32)
        tiny_scalar = jnp.asarray(1.0e-30, dtype=dtype)

        def _make_linearized_guess(attempt_context: _ThetaAttemptContext):
            flat_y = attempt_context.y
            t_value = attempt_context.t
            h_value = attempt_context.trial_dt
            f_old = attempt_context.f_old
            lagged_response = attempt_context.lagged_response
            t_new = attempt_context.t_new
            guess0 = _project_flat_state_if_needed(flat_y + h_value * f_old, project_flat)
            f_ref_new = flat_rhs(t_new, flat_y)
            jacobian_ref = jax.jacfwd(lambda y: flat_rhs(t_new, y))(flat_y)

            def _eval_new_rhs(y_value):
                y_proj = _project_flat_state_if_needed(y_value, project_flat)
                if lagged_response is not None:
                    return flat_rhs_with_lagged_response(t_new, y_proj, lagged_response)
                if use_lagged_linear_response:
                    return f_ref_new + jacobian_ref @ (y_proj - flat_y)
                return flat_rhs(t_new, y_proj)

            if predictor_mode == "euler":
                return guess0, jnp.asarray(True)

            jacobian_guess0 = jnp.where(
                use_lagged_linear_response,
                jacobian_ref,
                jax.jacfwd(lambda y: flat_rhs(t_new, y))(guess0),
            )
            system0 = identity_n - h_value * theta * jacobian_guess0
            lu0, piv0 = jax.scipy.linalg.lu_factor(system0)
            linearization_finite = jnp.logical_and(jnp.all(jnp.isfinite(system0)), jnp.all(jnp.isfinite(lu0)))

            def _corrector_body(_, carry):
                guess, linear_ok = carry
                guess_proj = _project_flat_state_if_needed(guess, project_flat)
                f_guess = _eval_new_rhs(guess_proj)
                affine_rhs = flat_y + h_value * (
                    (one - theta) * f_old
                    + theta * (f_guess - jacobian_guess0 @ guess_proj)
                )
                next_guess = jax.scipy.linalg.lu_solve((lu0, piv0), affine_rhs)
                finite = jnp.logical_and(jnp.all(jnp.isfinite(next_guess)), linearization_finite)
                next_guess = jnp.where(finite, next_guess, guess_proj)
                next_guess = _project_flat_state_if_needed(next_guess, project_flat)
                return next_guess, jnp.logical_and(linear_ok, finite)

            guess_pc, linear_ok_pc = jax.lax.fori_loop(
                0,
                n_linearized_solves,
                _corrector_body,
                (guess0, jnp.asarray(True)),
            )
            guess_fallback = _project_flat_state_if_needed(flat_y + h_value * f_old, project_flat)
            return (
                jnp.where(linear_ok_pc, guess_pc, guess_fallback),
                linear_ok_pc,
            )

        def _single_theta_newton_step(attempt_context: _ThetaAttemptContext):
            flat_y = attempt_context.y
            t_value = attempt_context.t
            h_value = attempt_context.trial_dt
            f_old = attempt_context.f_old
            lagged_response = attempt_context.lagged_response
            lagged_response_reused = attempt_context.lagged_response_reused
            reuse_state = attempt_context.reuse_state
            t_new = attempt_context.t_new
            guess0, linear_ok0 = _make_linearized_guess(attempt_context)
            f_ref_new = flat_rhs(t_new, flat_y)
            jacobian_ref = jax.jacfwd(lambda y: flat_rhs(t_new, y))(flat_y)

            def _eval_new_rhs(y_value):
                y_proj = _project_flat_state_if_needed(y_value, project_flat)
                if lagged_response is not None:
                    return flat_rhs_with_lagged_response(t_new, y_proj, lagged_response)
                if use_lagged_linear_response:
                    return f_ref_new + jacobian_ref @ (y_proj - flat_y)
                return flat_rhs(t_new, y_proj)

            def residual(y_val):
                y_proj = _project_flat_state_if_needed(y_val, project_flat)
                f_new = _eval_new_rhs(y_proj)
                return y_proj - flat_y - h_value * ((one - theta) * f_old + theta * f_new)

            jacobian_dt_scale = jnp.maximum(
                jnp.abs(reuse_state.cache_dt),
                jnp.asarray(1.0e-14, dtype=dtype),
            )
            dt_close = jnp.abs(h_value - reuse_state.cache_dt) <= jacobian_reuse_rtol * jacobian_dt_scale
            can_reuse_linearization = jnp.logical_and(
                jnp.logical_and(reuse_state.cache_valid, dt_close),
                jnp.logical_and(
                    reuse_state.cache_age < max_jacobian_age,
                    jnp.logical_not(use_lagged_linear_response),
                ),
            )

            def _reuse_linearization(_):
                return reuse_state.jacobian, reuse_state.lu_factor, reuse_state.pivots

            def _recompute_linearization(_):
                residual_jacobian_guess0 = jax.jacfwd(residual)(guess0)
                lu0, piv0 = jax.scipy.linalg.lu_factor(residual_jacobian_guess0)
                return residual_jacobian_guess0, lu0, piv0

            frozen_system, frozen_lu, frozen_piv = jax.lax.cond(
                can_reuse_linearization,
                _reuse_linearization,
                _recompute_linearization,
                operand=None,
            )
            frozen_linearization_finite = jnp.logical_and(
                jnp.all(jnp.isfinite(frozen_system)),
                jnp.all(jnp.isfinite(frozen_lu)),
            )

            def body_fn(carry):
                (
                    iter_idx,
                    y_cur,
                    residual_norm,
                    diverged,
                    prev_delta_norm,
                    theta_est,
                    final_delta_norm,
                    slow_contraction,
                    residual_blowup,
                    newton_nonfinite,
                ) = carry
                y_proj = _project_flat_state_if_needed(y_cur, project_flat)
                residual_cur = residual(y_proj)
                if freeze_attempt_linearization:
                    system = frozen_system
                    lu = frozen_lu
                    piv = frozen_piv
                    finite_system = frozen_linearization_finite
                else:
                    system = jax.jacfwd(residual)(y_proj)
                    lu, piv = jax.scipy.linalg.lu_factor(system)
                    finite_system = jnp.logical_and(jnp.all(jnp.isfinite(system)), jnp.all(jnp.isfinite(lu)))
                delta = jax.scipy.linalg.lu_solve((lu, piv), -residual_cur)
                delta = jnp.where(jnp.all(jnp.isfinite(delta)), delta, jnp.zeros_like(delta))
                delta = jnp.where(finite_system, delta, jnp.zeros_like(delta))
                delta_norm = jnp.linalg.norm(delta)
                theta_candidate = jnp.where(
                    iter_idx > 0,
                    delta_norm / jnp.maximum(prev_delta_norm, tiny_scalar),
                    theta_est,
                )
                theta_next = jnp.where(iter_idx > 0, theta_candidate, theta_est)
                base_norm = jnp.linalg.norm(residual_cur)
                accept_factor = jnp.asarray(0.999, dtype=dtype)

                def ls_cond(ls_state):
                    tau, cand_y, cand_norm, accepted = ls_state
                    need_more = jnp.logical_and(jnp.logical_not(accepted), tau > tau_min + tiny_scalar)
                    return need_more

                def ls_body(ls_state):
                    tau, _cand_y, cand_norm, accepted = ls_state
                    tau_next = jnp.maximum(tau * delta_reduction_factor, tau_min)
                    trial_y = _project_flat_state_if_needed(y_proj + tau_next * delta, project_flat)
                    trial_norm = jnp.linalg.norm(residual(trial_y))
                    accepted_next = jnp.logical_and(
                        jnp.isfinite(trial_norm),
                        trial_norm <= base_norm * accept_factor,
                    )
                    return tau_next, trial_y, trial_norm, accepted_next

                trial_y0 = _project_flat_state_if_needed(y_proj + delta, project_flat)
                trial_norm0 = jnp.linalg.norm(residual(trial_y0))
                accepted0 = jnp.logical_and(
                    finite_system,
                    jnp.logical_and(
                        jnp.isfinite(trial_norm0),
                        trial_norm0 <= base_norm * accept_factor,
                    ),
                )
                tau_final, y_next, residual_next, accepted_final = jax.lax.while_loop(
                    ls_cond,
                    ls_body,
                    (one, trial_y0, trial_norm0, accepted0),
                )
                nonfinite_state = jnp.logical_not(jnp.logical_and(jnp.all(jnp.isfinite(y_next)), jnp.isfinite(residual_next)))
                diverged_next = jnp.logical_or(diverged, jnp.logical_or(jnp.logical_not(accepted_final), nonfinite_state))
                residual_blowup_next = jnp.logical_or(
                    residual_blowup,
                    jnp.logical_and(jnp.isfinite(residual_next), residual_next > base_norm * jnp.asarray(1.5, dtype=dtype)),
                )
                newton_nonfinite_next = jnp.logical_or(
                    newton_nonfinite,
                    jnp.logical_not(
                        jnp.logical_and(
                            finite_system,
                            jnp.logical_and(jnp.all(jnp.isfinite(y_next)), jnp.isfinite(residual_next)),
                        )
                    ),
                )
                slow_contraction_next = jnp.logical_or(
                    slow_contraction,
                    jnp.logical_and(iter_idx > 0, theta_next >= jnp.asarray(0.8, dtype=dtype)),
                )
                return (
                    iter_idx + 1,
                    y_next,
                    residual_next,
                    diverged_next,
                    delta_norm,
                    theta_next,
                    delta_norm,
                    slow_contraction_next,
                    residual_blowup_next,
                    newton_nonfinite_next,
                )

            init_residual = residual(guess0)
            init_state = (
                jnp.asarray(0, dtype=jnp.int32),
                guess0,
                jnp.linalg.norm(init_residual),
                jnp.logical_not(linear_ok0),
                jnp.asarray(0.0, dtype=dtype),
                jnp.asarray(0.0, dtype=dtype),
                jnp.asarray(0.0, dtype=dtype),
                jnp.asarray(False),
                jnp.asarray(False),
                jnp.logical_not(jnp.logical_and(linear_ok0, jnp.all(jnp.isfinite(init_residual)))),
            )

            def cond_fn(carry):
                iter_idx, _y_cur, residual_norm, diverged, *_rest = carry
                active = residual_norm > self.tol
                return jnp.logical_and(jnp.logical_and(iter_idx < self.maxiter, active), jnp.logical_not(diverged))

            (
                iter_final,
                y_final,
                residual_norm_final,
                diverged_final,
                _prev_delta_norm_final,
                theta_final,
                final_delta_norm,
                slow_contraction_final,
                residual_blowup_final,
                newton_nonfinite_final,
            ) = jax.lax.while_loop(cond_fn, body_fn, init_state)
            converged = jnp.logical_and(
                jnp.logical_and(jnp.all(jnp.isfinite(y_final)), residual_norm_final <= self.tol),
                jnp.logical_not(diverged_final),
            )
            trial_y = _project_flat_state_if_needed(y_final, project_flat)
            return _ThetaAcceptedStepAttemptResult(
                trial_dt=h_value,
                trial_y=trial_y,
                err_norm=residual_norm_final,
                converged=converged,
                newton_iter_count=iter_final,
                final_residual_norm=residual_norm_final,
                final_delta_norm=final_delta_norm,
                theta_final=theta_final,
                slow_contraction=slow_contraction_final,
                residual_blowup=residual_blowup_final,
                newton_nonfinite=newton_nonfinite_final,
                lagged_reused=lagged_response_reused,
                jacobian_reused=can_reuse_linearization,
                lagged_response_cache_out=lagged_response,
                lagged_response_valid_out=jnp.asarray(
                    use_transport_lagged_response and (lagged_response_reuse_mode == "global_state_drift")
                ),
                lagged_reference_y_out=flat_y,
                jacobian_out=frozen_system,
                cache_valid_out=jnp.asarray(True),
                cache_dt_out=h_value,
                cache_age_out=jnp.where(can_reuse_linearization, reuse_state.cache_age + 1, jnp.asarray(0, dtype=jnp.int32)),
                lu_out=frozen_lu,
                piv_out=frozen_piv,
                diverged_final=diverged_final,
                nonfinite_stage_state=jnp.logical_not(jnp.all(jnp.isfinite(trial_y))),
                nonfinite_stage_residual=jnp.logical_not(jnp.isfinite(residual_norm_final)),
                finite_f0=jnp.all(jnp.isfinite(f_old)),
                finite_z0=jnp.all(jnp.isfinite(flat_y)),
                finite_initial_residual=jnp.all(jnp.isfinite(init_residual)),
            )

        def step_fn(step_state: _ThetaStepState, _):
            failed = step_state.status[STATUS_FAILED] != 0
            fail_code = step_state.status[STATUS_FAIL_CODE]
            n_accepted = step_state.status[STATUS_N_ACCEPTED]

            def _skip(_):
                return step_state, _ThetaStepInfo(
                    y=step_state.y,
                    t=step_state.t,
                    dt=jnp.asarray(0.0, dtype=dtype),
                    accepted=jnp.asarray(False),
                    failed=failed,
                    fail_code=fail_code,
                )

            def _run(_):
                attempt_context0 = _theta_make_attempt_context(
                    step_state,
                    t_final=t_final,
                    flat_rhs=flat_rhs,
                    build_lagged_response=build_lagged_response,
                    unpack_flat=unpack_flat,
                    project_flat=project_flat,
                    use_transport_lagged_response=use_transport_lagged_response,
                )
                lagged_response0, lagged_reference_y0, lagged_response_reused0 = _theta_prepare_lagged_response(
                    step_state,
                    use_transport_lagged_response=use_transport_lagged_response,
                    lagged_response_reuse_mode=lagged_response_reuse_mode,
                    lagged_response_reuse_rtol=lagged_response_reuse_rtol,
                    lagged_response_reuse_atol=lagged_response_reuse_atol,
                    unpack_flat=unpack_flat,
                    project_flat=project_flat,
                    build_lagged_response=build_lagged_response,
                )
                attempt_context0 = dataclasses.replace(
                    attempt_context0,
                    lagged_response=lagged_response0,
                    lagged_reference_y=lagged_reference_y0,
                    lagged_response_reused=lagged_response_reused0,
                )
                trial_dt0 = attempt_context0.trial_dt

                def retry_cond(carry):
                    trial_dt, _trial_y, converged, *_rest = carry
                    can_reduce = trial_dt > dt_min * (jnp.asarray(1.0, dtype=dtype) + jnp.asarray(1.0e-12, dtype=dtype))
                    return jnp.logical_and(jnp.logical_not(converged), can_reduce)

                def retry_body(carry):
                    trial_dt, _trial_y, _converged, *_rest = carry
                    reduced_dt = jnp.maximum(trial_dt * delta_reduction_factor, dt_min)
                    retry_context = _theta_attempt_context_with_dt(attempt_context0, reduced_dt)
                    next_attempt = _single_theta_newton_step(retry_context)
                    return (
                        reduced_dt,
                        next_attempt.trial_y,
                        next_attempt.converged,
                        next_attempt.newton_iter_count,
                        next_attempt.err_norm,
                        next_attempt.final_delta_norm,
                        next_attempt.theta_final,
                        next_attempt.slow_contraction,
                        next_attempt.residual_blowup,
                        next_attempt.newton_nonfinite,
                        next_attempt.finite_initial_residual,
                        next_attempt.lagged_reused,
                        next_attempt.jacobian_reused,
                        next_attempt.lagged_response_cache_out,
                        next_attempt.lagged_response_valid_out,
                        next_attempt.lagged_reference_y_out,
                        next_attempt.jacobian_out,
                        next_attempt.cache_valid_out,
                        next_attempt.cache_dt_out,
                        next_attempt.cache_age_out,
                        next_attempt.lu_out,
                        next_attempt.piv_out,
                    )

                attempt0 = _single_theta_newton_step(attempt_context0)
                (
                    trial_dt,
                    trial_y,
                    converged,
                    iter_count,
                    residual_norm,
                    final_delta_norm,
                    theta_final,
                    slow_contraction,
                    residual_blowup,
                    newton_nonfinite,
                    finite_initial_residual,
                    lagged_reused,
                    jacobian_reused,
                    lagged_response_cache_out,
                    lagged_response_valid_out,
                    lagged_reference_y_out,
                    jacobian_out,
                    cache_valid_out,
                    cache_dt_out,
                    cache_age_out,
                    lu_out,
                    piv_out,
                ) = jax.lax.while_loop(
                    retry_cond,
                    retry_body,
                    (
                        trial_dt0,
                        attempt0.trial_y,
                        attempt0.converged,
                        attempt0.newton_iter_count,
                        attempt0.err_norm,
                        attempt0.final_delta_norm,
                        attempt0.theta_final,
                        attempt0.slow_contraction,
                        attempt0.residual_blowup,
                        attempt0.newton_nonfinite,
                        attempt0.finite_initial_residual,
                        attempt0.lagged_reused,
                        attempt0.jacobian_reused,
                        attempt0.lagged_response_cache_out,
                        attempt0.lagged_response_valid_out,
                        attempt0.lagged_reference_y_out,
                        attempt0.jacobian_out,
                        attempt0.cache_valid_out,
                        attempt0.cache_dt_out,
                        attempt0.cache_age_out,
                        attempt0.lu_out,
                        attempt0.piv_out,
                    ),
                )
                attempt_result = _ThetaAcceptedStepAttemptResult(
                    trial_dt=trial_dt,
                    trial_y=trial_y,
                    err_norm=residual_norm,
                    converged=converged,
                    newton_iter_count=iter_count,
                    final_residual_norm=residual_norm,
                    final_delta_norm=final_delta_norm,
                    theta_final=theta_final,
                    slow_contraction=slow_contraction,
                    residual_blowup=residual_blowup,
                    newton_nonfinite=newton_nonfinite,
                    lagged_reused=lagged_reused,
                    jacobian_reused=jacobian_reused,
                    lagged_response_cache_out=lagged_response_cache_out,
                    lagged_response_valid_out=lagged_response_valid_out,
                    lagged_reference_y_out=lagged_reference_y_out,
                    jacobian_out=jacobian_out,
                    cache_valid_out=cache_valid_out,
                    cache_dt_out=cache_dt_out,
                    cache_age_out=cache_age_out,
                    lu_out=lu_out,
                    piv_out=piv_out,
                    diverged_final=jnp.logical_not(converged),
                    nonfinite_stage_state=jnp.logical_not(jnp.all(jnp.isfinite(trial_y))),
                    nonfinite_stage_residual=jnp.logical_not(jnp.isfinite(residual_norm)),
                    finite_f0=jnp.all(jnp.isfinite(attempt_context0.f_old)),
                    finite_z0=jnp.all(jnp.isfinite(step_state.y)),
                    finite_initial_residual=finite_initial_residual,
                )
                controller_update = _theta_controller_update(
                    step_state=step_state,
                    trial_dt=attempt_result.trial_dt,
                    converged=attempt_result.converged,
                    residual_norm=attempt_result.err_norm,
                    newton_iter_count=attempt_result.newton_iter_count,
                    theta_final=attempt_result.theta_final,
                    slow_contraction=attempt_result.slow_contraction,
                    lagged_reused=attempt_result.lagged_reused,
                    jacobian_reused=attempt_result.jacobian_reused,
                    dt_min=dt_min,
                    dt_max=dt_max,
                    safety_factor=safety_factor,
                    min_step_factor=min_step_factor,
                    max_step_factor=max_step_factor,
                    controller_mode=controller_mode,
                    dtype=dtype,
                )
                attempt_reuse_state = _theta_make_reuse_state(
                    lagged_response_cache=attempt_result.lagged_response_cache_out,
                    lagged_response_available=jnp.asarray(attempt_context0.lagged_response is not None),
                    lagged_response_valid=attempt_result.lagged_response_valid_out,
                    lagged_reference_y=attempt_result.lagged_reference_y_out,
                    jacobian=attempt_result.jacobian_out,
                    cache_valid=attempt_result.cache_valid_out,
                    cache_dt=attempt_result.cache_dt_out,
                    cache_age=attempt_result.cache_age_out,
                    lu_factor=attempt_result.lu_out,
                    pivots=attempt_result.piv_out,
                    freeze_attempt_linearization=jnp.asarray(freeze_attempt_linearization),
                    last_lagged_reused=attempt_result.lagged_reused,
                    last_jacobian_reused=attempt_result.jacobian_reused,
                    last_linearization_dt=attempt_result.trial_dt,
                )
                return _theta_step_transition_from_attempt(
                    step_state,
                    attempt_result=attempt_result,
                    n_accepted=n_accepted,
                    next_dt_if_accepted=controller_update["next_dt_accept"],
                    next_dt_if_rejected=controller_update["next_dt_reject"],
                    next_prev_error_if_accepted=controller_update["safe_error"],
                    next_prev_dt_if_accepted=attempt_result.trial_dt,
                    next_recent_reject_count_if_accepted=jnp.asarray(0, dtype=jnp.int32),
                    next_recent_reject_count_if_rejected=controller_update["retry_count_next"],
                    next_regrowth_cooldown_if_accepted=controller_update["regrowth_cooldown_next"],
                    next_regrowth_cooldown_if_rejected=controller_update["reject_regrowth_cooldown_next"],
                    next_easy_growth_streak_if_accepted=controller_update["easy_growth_streak_next"],
                    next_easy_growth_streak_if_rejected=jnp.asarray(0, dtype=jnp.int32),
                    next_reuse_state_if_accepted=attempt_reuse_state,
                    next_reuse_state_if_rejected=attempt_reuse_state,
                    project_flat=project_flat,
                    dtype=dtype,
                )

            return jax.lax.cond(failed, _skip, _run, operand=None)

        step_state0 = _ThetaStepState(
            t=t0,
            y=flat_state0,
            dt=base_dt,
            status=jnp.asarray([0, 0, 0], dtype=jnp.int32),
            prev_error=jnp.asarray(1.0, dtype=dtype),
            prev_dt=jnp.asarray(0.0, dtype=dtype),
            recent_reject_count=jnp.asarray(0, dtype=jnp.int32),
            regrowth_cooldown=jnp.asarray(0, dtype=jnp.int32),
            easy_growth_streak=jnp.asarray(0, dtype=jnp.int32),
            prev_theta_final=jnp.asarray(0.0, dtype=dtype),
            prev_newton_iter_count=jnp.asarray(0, dtype=jnp.int32),
            reuse_state=_theta_initial_reuse_state(state_dim, dtype),
        )
        save_n = getattr(self, "save_n", None)
        save_n = max(1, int(save_n)) if save_n is not None else 1
        stop_after_accepted_steps = getattr(self, "stop_after_accepted_steps", None)
        (
            step_state_f,
            _,
            _,
            ys_saved,
            ts_saved,
            dts_saved,
            accepted_mask_saved,
            failed_mask_saved,
            fail_codes_saved,
            last_attempt_accepted,
            last_attempt_converged,
            last_attempt_err_norm,
            last_attempt_fail_code,
            last_attempt_diverged,
            last_attempt_nonfinite_stage_state,
            last_attempt_nonfinite_stage_residual,
            last_attempt_finite_f0,
            last_attempt_finite_z0,
            last_attempt_finite_initial_residual,
            last_attempt_newton_iter_count,
            last_attempt_final_residual_norm,
            last_attempt_final_delta_norm,
            last_attempt_theta_final,
            last_attempt_slow_contraction,
            last_attempt_residual_blowup,
            last_attempt_newton_nonfinite,
            last_attempt_lagged_reused,
            last_attempt_jacobian_reused,
        ) = _run_saved_loop(
            step_state0=step_state0,
            step_fn=step_fn,
            save_n=save_n,
            t0=t0,
            t_final=t_final,
            state_dim=state_dim,
            dtype=dtype,
            max_total_steps=max_total_steps,
            stop_after_accepted_steps=stop_after_accepted_steps,
        )
        failed_f = step_state_f.status[STATUS_FAILED] != 0
        fail_code_f = step_state_f.status[STATUS_FAIL_CODE]
        n_acc_f = step_state_f.status[STATUS_N_ACCEPTED]
        accepted_limit_hit = _accepted_step_limit_reached(step_state_f, stop_after_accepted_steps)
        return _finalize_custom_solver_output(
            ys_saved,
            ts_saved,
            dts_saved,
            accepted_mask_saved,
            failed_mask_saved,
            fail_codes_saved,
            step_state_f.y,
            step_state_f.t,
            jnp.logical_or(step_state_f.t >= (t_final - 1.0e-15), accepted_limit_hit),
            failed_f,
            fail_code_f,
            n_acc_f,
            last_attempt_accepted,
            last_attempt_converged,
            last_attempt_err_norm,
            last_attempt_fail_code,
            last_attempt_diverged,
            last_attempt_nonfinite_stage_state,
            last_attempt_nonfinite_stage_residual,
            last_attempt_finite_f0,
            last_attempt_finite_z0,
            last_attempt_finite_initial_residual,
            last_attempt_newton_iter_count,
            last_attempt_final_residual_norm,
            last_attempt_final_delta_norm,
            last_attempt_theta_final,
            last_attempt_slow_contraction,
            last_attempt_residual_blowup,
            last_attempt_newton_nonfinite,
            last_attempt_lagged_reused,
            last_attempt_jacobian_reused,
            step_state_f.reuse_state,
            unpack_flat,
            state,
            species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
            density_floor=density_floor,
            temperature_floor=temperature_floor,
        )

def build_time_solver(solver_parameters: Any, solver_override: Any = None) -> TransportSolver:
    """Create a time solver backend from runtime parameters/config.

    `solver_override` can be either:
      - an instance with `.solve(...)` (used directly), or
      - a diffrax solver instance (wrapped in DiffraxSolver).
    """
    import diffrax

    def _cfg_get(key: str, default=None):
        if isinstance(solver_parameters, dict):
            return solver_parameters.get(key, default)
        return getattr(solver_parameters, key, default)

    t0 = float(_cfg_get("t0"))
    t1 = float(_cfg_get("t_final", _cfg_get("t1")))
    dt = float(_cfg_get("dt"))

    if solver_override is not None:
        if hasattr(solver_override, "solve"):
            return solver_override
        return DiffraxSolver(integrator=lambda: solver_override, t0=t0, t1=t1, dt=dt)

    _, backend = _select_solver_family_and_backend(solver_parameters)
    save_n = _cfg_get("save_n", _cfg_get("n_save"))
    generic_rhs_mode = _cfg_get("rhs_mode", "black_box")
    stop_after_accepted_steps = _cfg_get("stop_after_accepted_steps")
    if backend == "theta":
        return ThetaMethodSolver(
            t0=t0,
            t1=t1,
            dt=dt,
            min_step=float(_cfg_get("min_step", 1.0e-14)),
            theta_implicit=float(_cfg_get("theta_implicit", 1.0)),
            predictor_mode=str(_cfg_get("theta_predictor_mode", "linearized")),
            rhs_mode=str(_cfg_get("theta_rhs_mode", generic_rhs_mode)),
            use_predictor_corrector=bool(_cfg_get("use_predictor_corrector", False)),
            n_corrector_steps=int(_cfg_get("n_corrector_steps", 1)),
            tol=float(_cfg_get("nonlinear_solver_tol", _cfg_get("tol", 1.0e-8))),
            max_steps=int(_cfg_get("max_steps", 20000)),
            stop_after_accepted_steps=stop_after_accepted_steps,
            save_n=save_n,
        )
    if backend == "theta_newton":
        return NewtonThetaMethodSolver(
            t0=t0,
            t1=t1,
            dt=dt,
            min_step=float(_cfg_get("min_step", 1.0e-14)),
            theta_implicit=float(_cfg_get("theta_implicit", 1.0)),
            predictor_mode=str(_cfg_get("theta_predictor_mode", "linearized")),
            rhs_mode=str(_cfg_get("theta_rhs_mode", generic_rhs_mode)),
            use_predictor_corrector=bool(_cfg_get("use_predictor_corrector", False)),
            n_corrector_steps=int(_cfg_get("n_corrector_steps", 1)),
            tol=float(_cfg_get("nonlinear_solver_tol", _cfg_get("tol", 1.0e-8))),
            maxiter=int(_cfg_get("nonlinear_solver_maxiter", _cfg_get("maxiter", 20))),
            max_step=float(_cfg_get("max_step", max(t1 - t0, dt))),
            safety_factor=float(_cfg_get("safety_factor", 0.9)),
            min_step_factor=float(_cfg_get("min_step_factor", 0.5)),
            max_step_factor=float(_cfg_get("max_step_factor", 2.0)),
            controller_mode=str(_cfg_get("theta_controller_mode", "current")),
            jacobian_reuse_mode=str(_cfg_get("theta_jacobian_reuse_mode", "refresh_each_iteration")),
            jacobian_reuse_rtol=float(_cfg_get("theta_jacobian_reuse_rtol", 0.1)),
            max_jacobian_age=int(_cfg_get("theta_max_jacobian_age", 8)),
            lagged_response_reuse_mode=str(_cfg_get("theta_lagged_response_reuse_mode", "retry_only")),
            lagged_response_reuse_rtol=float(_cfg_get("theta_lagged_response_reuse_rtol", 5.0e-2)),
            lagged_response_reuse_atol=float(_cfg_get("theta_lagged_response_reuse_atol", 1.0e-8)),
            target_nonlinear_iterations=int(_cfg_get("theta_target_nonlinear_iterations", 4)),
            delta_reduction_factor=float(_cfg_get("theta_delta_reduction_factor", 0.5)),
            tau_min=float(_cfg_get("theta_tau_min", 0.01)),
            max_steps=int(_cfg_get("max_steps", 20000)),
            stop_after_accepted_steps=stop_after_accepted_steps,
            save_n=save_n,
        )
    if backend == "radau":
        return RADAUSolver(
            t0=t0,
            t1=t1,
            dt=dt,
            rtol=float(_cfg_get("rtol", 1.0e-6)),
            atol=float(_cfg_get("atol", 1.0e-8)),
            max_step=float(_cfg_get("max_step", max(t1 - t0, dt))),
            min_step=float(_cfg_get("min_step", 1.0e-14)),
            tol=float(_cfg_get("nonlinear_solver_tol", _cfg_get("tol", 1.0e-8))),
            maxiter=int(_cfg_get("nonlinear_solver_maxiter", _cfg_get("maxiter", 20))),
            error_estimator=str(_cfg_get("radau_error_estimator", "embedded2")),
            num_stages=int(_cfg_get("radau_num_stages", 3)),
            safety_factor=float(_cfg_get("safety_factor", 0.9)),
            min_step_factor=float(_cfg_get("min_step_factor", 0.1)),
            max_step_factor=float(_cfg_get("max_step_factor", 5.0)),
            rhs_mode=str(_cfg_get("radau_rhs_mode", generic_rhs_mode)),
            newton_divergence_mode=str(_cfg_get("radau_newton_divergence_mode", "legacy")),
            newton_residual_norm=str(_cfg_get("radau_newton_residual_norm", "raw")),
            newton_tol_mode=str(_cfg_get("radau_newton_tol_mode", "residual")),
            newton_fnewt_mode=str(_cfg_get("radau_newton_fnewt_mode", "tol")),
            controller_mode=str(_cfg_get("radau_controller_mode", "current")),
            predictor_mode=str(_cfg_get("radau_predictor_mode", "current")),
            lagged_response_reuse_mode=str(_cfg_get("lagged_response_reuse_mode", "retry_only")),
            lagged_response_reuse_rtol=float(_cfg_get("lagged_response_reuse_rtol", 5.0e-2)),
            lagged_response_reuse_atol=float(_cfg_get("lagged_response_reuse_atol", 1.0e-8)),
            max_steps=int(_cfg_get("max_steps", 20000)),
            stop_after_accepted_steps=stop_after_accepted_steps,
            debug_stage_markers=bool(_cfg_get("debug_stage_markers", False)),
            debug_walltime_attempts=bool(_cfg_get("debug_walltime_attempts", False)),
            save_n=save_n,
        )
    integrator_ctor = _get_diffrax_integrator(backend)
    ts_list = _cfg_get("ts_list")
    saveat = diffrax.SaveAt(ts=ts_list) if ts_list is not None else diffrax.SaveAt(t1=True)
    stepsize_controller = diffrax.PIDController(
        pcoeff=0.3,
        icoeff=0.4,
        rtol=float(_cfg_get("rtol")),
        atol=float(_cfg_get("atol")),
    )
    return DiffraxSolver(
        integrator=integrator_ctor,
        t0=t0,
        t1=t1,
        dt=dt,
        save_n=save_n,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=int(_cfg_get("max_steps", 20000)),
        throw=bool(_cfg_get("throw", False)),
    )

register_time_solver("diffrax_kvaerno5", lambda **kw: DiffraxSolver(_get_diffrax_integrator("diffrax_kvaerno5"), **kw))
register_time_solver("diffrax_tsit5", lambda **kw: DiffraxSolver(_get_diffrax_integrator("diffrax_tsit5"), **kw))
register_time_solver("diffrax_dopri5", lambda **kw: DiffraxSolver(_get_diffrax_integrator("diffrax_dopri5"), **kw))
register_time_solver("radau", lambda **kw: RADAUSolver(**kw))
register_time_solver("theta", lambda **kw: ThetaMethodSolver(**kw))
register_time_solver("theta_newton", lambda **kw: NewtonThetaMethodSolver(**kw))
