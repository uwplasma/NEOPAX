
# transport_solvers.py
"""
Modular solver interface and backends for NEOPAX transport equations.
Supports multiple solver backends: time integration, root-finding, and optimization.
Inspired by torax (https://github.com/google-deepmind/torax).
"""

from typing import Callable, Any
import dataclasses
import inspect
import jax
import jax.numpy as jnp
import numpy as np


TIME_SOLVER_REGISTRY: dict[str, Callable[..., "TransportSolver"]] = {}

ODE_SOLVER_BACKENDS = {
    "diffrax_kvaerno5",
    "diffrax_tsit5",
    "diffrax_dopri5",
    "radau",
    "rosenbrock",
}


def _build_radau3_transform_constants() -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Return real stage transform data for the 3-stage Radau IIA tableau."""
    sqrt6 = np.sqrt(6.0)
    a = np.asarray(
        [
            [(88.0 - 7.0 * sqrt6) / 360.0, (296.0 - 169.0 * sqrt6) / 1800.0, (-2.0 + 3.0 * sqrt6) / 225.0],
            [(296.0 + 169.0 * sqrt6) / 1800.0, (88.0 + 7.0 * sqrt6) / 360.0, (-2.0 - 3.0 * sqrt6) / 225.0],
            [(16.0 - sqrt6) / 36.0, (16.0 + sqrt6) / 36.0, 1.0 / 9.0],
        ],
        dtype=np.float64,
    )
    eigvals, eigvecs = np.linalg.eig(a)
    real_idx = int(np.argmin(np.abs(eigvals.imag)))
    complex_candidates = [idx for idx in range(3) if idx != real_idx]
    complex_idx = max(complex_candidates, key=lambda idx: eigvals[idx].imag)
    real_vec = eigvecs[:, real_idx].real
    complex_vec = eigvecs[:, complex_idx]
    transform = np.column_stack([real_vec, complex_vec.real, complex_vec.imag]).astype(np.float64)
    inv_transform = np.linalg.inv(transform)
    real_eig = float(eigvals[real_idx].real)
    complex_block = np.asarray(
        [
            [float(eigvals[complex_idx].real), float(eigvals[complex_idx].imag)],
            [-float(eigvals[complex_idx].imag), float(eigvals[complex_idx].real)],
        ],
        dtype=np.float64,
    )
    return transform, inv_transform, real_eig, complex_block


_RADAU3_TRANSFORM, _RADAU3_INV_TRANSFORM, _RADAU3_REAL_EIG, _RADAU3_COMPLEX_BLOCK = _build_radau3_transform_constants()


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
    backend = str(
        solver_parameters.get(
            "transport_solver_backend",
            solver_parameters.get("integrator", "diffrax_kvaerno5"),
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
            )
        rebuilt = dataclasses.replace(template_state, density=density, pressure=pressure, Er=er)
        return _project_fixed_temperature_output(
            rebuilt,
            template_state,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
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
) -> Any:
    from ._transport_equations import project_fixed_temperature_species

    density = getattr(state_like, "density", None)
    ref_density = getattr(reference_state, "density", None)
    if density is None or ref_density is None:
        return state_like

    if density.ndim == ref_density.ndim + 1:
        out = jax.vmap(
            lambda s: project_fixed_temperature_species(
                s,
                temperature_active_mask=temperature_active_mask,
                fixed_temperature_profile=fixed_temperature_profile,
            )
        )(state_like)
        return _restore_state_metadata(out, reference_state)
    out = project_fixed_temperature_species(
        state_like,
        temperature_active_mask=temperature_active_mask,
        fixed_temperature_profile=fixed_temperature_profile,
    )
    return _restore_state_metadata(out, reference_state)


def _apply_quasi_neutrality_output(
    state_like: Any,
    species: Any,
    reference_state: Any,
    temperature_active_mask: Any = None,
    fixed_temperature_profile: Any = None,
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
        )
    out = enforce_quasi_neutrality(state_like, species)
    return _project_fixed_temperature_output(
        out,
        reference_state,
        temperature_active_mask=temperature_active_mask,
        fixed_temperature_profile=fixed_temperature_profile,
    )


def _project_state_to_quasi_neutrality(
    state_like: Any,
    species: Any,
    temperature_active_mask: Any = None,
    fixed_temperature_profile: Any = None,
) -> Any:
    from ._transport_equations import enforce_quasi_neutrality, project_fixed_temperature_species

    if species is None:
        return project_fixed_temperature_species(
            state_like,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
        )
    density = getattr(state_like, "density", None)
    if density is None:
        return project_fixed_temperature_species(
            state_like,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
        )
    projected = enforce_quasi_neutrality(state_like, species)
    return project_fixed_temperature_species(
        projected,
        temperature_active_mask=temperature_active_mask,
        fixed_temperature_profile=fixed_temperature_profile,
    )


def _project_packed_transport_state_arrays(
    state_like: Any,
    template_state: Any,
    species: Any,
    temperature_active_mask: Any = None,
    fixed_temperature_profile: Any = None,
) -> Any:
    """Project packed solver arrays without rebuilding a full TransportState."""
    if not (isinstance(state_like, tuple) and len(state_like) == 3):
        return state_like

    density, pressure, er = state_like
    if temperature_active_mask is None or fixed_temperature_profile is None:
        return state_like

    eidx = _electron_density_index(species)
    if eidx is not None and density.shape[-2] == template_state.density.shape[0] - 1:
        full_shape = pressure.shape[:-2] + (template_state.density.shape[0], pressure.shape[-1])
        full_density = jnp.zeros(full_shape, dtype=pressure.dtype)
        full_density = full_density.at[..., :eidx, :].set(density[..., :eidx, :])
        full_density = full_density.at[..., eidx + 1 :, :].set(density[..., eidx:, :])
        charge_qp = jnp.asarray(species.charge_qp, dtype=pressure.dtype)
        ion_indices = jnp.asarray(getattr(species, "ion_indices", ()), dtype=int)
        if ion_indices.size > 0:
            Z_i = jnp.take(charge_qp, ion_indices, axis=0)
            n_i = jnp.take(full_density, ion_indices, axis=-2)
            n_e = -jnp.sum(Z_i[..., None] * n_i, axis=-2) / charge_qp[int(eidx)]
            full_density = full_density.at[..., int(eidx), :].set(n_e)
    else:
        full_density = density

    active_mask = jnp.asarray(temperature_active_mask, dtype=bool)
    fixed_temperature = jnp.asarray(fixed_temperature_profile, dtype=pressure.dtype)
    active_mask = active_mask.reshape((1,) * (pressure.ndim - 2) + (active_mask.shape[0], 1))
    fixed_temperature = jnp.broadcast_to(fixed_temperature, pressure.shape)
    fixed_pressure = full_density * fixed_temperature
    projected_pressure = jnp.where(active_mask, pressure, fixed_pressure)
    return (density, projected_pressure, er)


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
        )

    def unpack_packed(packed_state_like):
        return _unpack_transport_state_arrays(
            packed_state_like,
            template_state,
            species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
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
        state = _project_state_to_quasi_neutrality(
            state,
            species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
        )
        flat_state0, unpack_flat, _unpack_packed, _pack_state, project_flat = _make_solver_state_transform(
            state,
            species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
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
                    ),
                )
        return sol


def _tree_add(a: Any, b: Any) -> Any:
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


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
    unpack_flat,
    reference_state,
    species,
    temperature_active_mask=None,
    fixed_temperature_profile=None,
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
    )
    final_state = unpack_flat(y_final_flat)
    if species is not None:
        final_state = enforce_quasi_neutrality(final_state, species)
    final_state = _project_fixed_temperature_output(
        final_state,
        reference_state,
        temperature_active_mask=temperature_active_mask,
        fixed_temperature_profile=fixed_temperature_profile,
    )
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


def _solve_linear_system(
    matvec: Callable[[jax.Array], jax.Array],
    rhs: jax.Array,
    linear_solver: str,
    dense_matrix_builder: Callable[[], jax.Array] | None,
    gmres_tol: float,
    gmres_maxiter: int,
):
    if linear_solver == "gmres":
        sol, _ = jax.scipy.sparse.linalg.gmres(
            matvec,
            rhs,
            tol=gmres_tol,
            atol=gmres_tol,
            maxiter=gmres_maxiter,
        )
        return sol
    if dense_matrix_builder is None:
        raise ValueError("dense_matrix_builder must be provided for direct solves.")
    matrix = dense_matrix_builder()
    return jax.scipy.linalg.solve(matrix, rhs)


def _prefer_dense_direct(linear_solver: str, system_size: int, threshold: int = 2048) -> bool:
    mode = str(linear_solver).strip().lower()
    if mode in ("direct", "dense"):
        return True
    if mode == "auto":
        return system_size <= threshold
    return False


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class RosenbrockSolver(TransportSolver):
    t0: float
    t1: float
    dt: float
    rtol: float = 1.0e-6
    atol: float = 1.0e-8
    max_step: float = 1.0
    min_step: float = 1.0e-14
    linear_solver: str = "gmres"
    gmres_tol: float = 1.0e-8
    gmres_maxiter: int = 200
    safety_factor: float = 0.9
    min_step_factor: float = 0.2
    max_step_factor: float = 5.0
    gamma: float = 1.7071067811865475
    n_steps: int = 0

    def __init__(
        self,
        t0: float = 0.0,
        t1: float = 1.0,
        dt: float = 1.0e-2,
        rtol: float = 1.0e-6,
        atol: float = 1.0e-8,
        max_step: float = 1.0,
        min_step: float = 1.0e-14,
        linear_solver: str = "gmres",
        gmres_tol: float = 1.0e-8,
        gmres_maxiter: int = 200,
        safety_factor: float = 0.9,
        min_step_factor: float = 0.2,
        max_step_factor: float = 5.0,
        gamma: float = 1.7071067811865475,
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
        object.__setattr__(self, "linear_solver", str(linear_solver).strip().lower())
        object.__setattr__(self, "gmres_tol", float(gmres_tol))
        object.__setattr__(self, "gmres_maxiter", int(max(1, gmres_maxiter)))
        object.__setattr__(self, "safety_factor", float(safety_factor))
        object.__setattr__(self, "min_step_factor", float(min_step_factor))
        object.__setattr__(self, "max_step_factor", float(max_step_factor))
        object.__setattr__(self, "gamma", float(gamma))
        object.__setattr__(self, "n_steps", n_steps)
        object.__setattr__(self, "save_n", save_n)

    def solve(self, state, vector_field: Callable, *args, **kwargs):
        if self.newton_strategy == "simplified" and self.linear_solver != "gmres":
            return self._solve_standard(state, vector_field, *args, **kwargs)
        return self._solve_legacy(state, vector_field, *args, **kwargs)

    def _solve_legacy(self, state, vector_field: Callable, *args, **kwargs):
        species = _extract_species_from_args(args)
        temperature_active_mask, fixed_temperature_profile = _extract_fixed_temperature_projection(vector_field)
        state = _project_state_to_quasi_neutrality(
            state,
            species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
        )
        flat_state0, unpack_flat, _unpack_packed, _pack_state, project_flat = _make_solver_state_transform(
            state,
            species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
        )
        dtype = flat_state0.dtype
        t0 = jnp.asarray(self.t0, dtype=dtype)
        t_final = jnp.asarray(self.t1, dtype=dtype)
        dt_min = jnp.asarray(self.min_step, dtype=dtype)
        dt_max = jnp.asarray(self.max_step, dtype=dtype)
        base_dt = jnp.clip(jnp.asarray(self.dt, dtype=dtype), dt_min, dt_max)
        max_total_steps = int(max(16, self.n_steps * 32))
        gamma = jnp.asarray(self.gamma, dtype=dtype)
        flat_rhs = _flat_rhs_factory(unpack_flat, vector_field, args, kwargs, project_flat=project_flat)
        state_dim = flat_state0.shape[0]

        # L-stable ROS2 (Kaps-Rentrop / Hairer-Wanner-style) coefficients.
        a21 = 1.0 / gamma
        c21 = -2.0 / gamma
        m1 = 3.0 / (2.0 * gamma)
        m2 = 1.0 / (2.0 * gamma)
        e1 = 1.0 / (2.0 * gamma)
        e2 = 1.0 / (2.0 * gamma)

        def _error_norm(err_vec, flat_ref, flat_candidate):
            scale = self.atol + self.rtol * jnp.maximum(jnp.abs(flat_ref), jnp.abs(flat_candidate))
            normalized = err_vec / scale
            return jnp.sqrt(jnp.mean(normalized * normalized) + 1.0e-30)

        def _single_step(flat_y, t_value, h_value):
            f_n = flat_rhs(t_value, flat_y)
            use_dense_direct = _prefer_dense_direct(self.linear_solver, state_dim)
            linear_solver_mode = "direct" if use_dense_direct else self.linear_solver

            if use_dense_direct:
                jacobian = jax.jacfwd(lambda y: flat_rhs(t_value, y))(flat_y)
                system_matrix = jnp.eye(state_dim, dtype=dtype) - gamma * h_value * jacobian

                def matvec(v):
                    return system_matrix @ v

                dense_builder = lambda: system_matrix
                lin = lambda v: jacobian @ v
            else:
                _, lin = jax.linearize(lambda y: flat_rhs(t_value, y), flat_y)

                def matvec(v):
                    return v - gamma * h_value * lin(v)

                dense_builder = None

            rhs1 = h_value * f_n
            k1 = _solve_linear_system(
                matvec=matvec,
                rhs=rhs1,
                linear_solver=linear_solver_mode,
                dense_matrix_builder=dense_builder,
                gmres_tol=self.gmres_tol,
                gmres_maxiter=self.gmres_maxiter,
            )

            y_stage = flat_y + a21 * k1
            f_stage = flat_rhs(t_value + h_value, y_stage)
            rhs2 = h_value * f_stage + c21 * k1
            k2 = _solve_linear_system(
                matvec=matvec,
                rhs=rhs2,
                linear_solver=linear_solver_mode,
                dense_matrix_builder=dense_builder,
                gmres_tol=self.gmres_tol,
                gmres_maxiter=self.gmres_maxiter,
            )

            flat_next = flat_y + m1 * k1 + m2 * k2
            err_vec = e1 * k1 + e2 * k2
            err_norm = _error_norm(err_vec, flat_y, flat_next)
            success = jnp.logical_and(
                jnp.all(jnp.isfinite(flat_next)),
                jnp.all(jnp.isfinite(err_vec)),
            )
            return flat_next, err_norm, success

        def _attempt_step(carry):
            t_value, flat_y, dt_value, done, failed, fail_code, n_accepted = carry
            trial_dt = jnp.minimum(dt_value, t_final - t_value)
            trial_y, err_norm, converged = _single_step(flat_y, t_value, trial_dt)
            accepted = jnp.logical_and(converged, err_norm <= 1.0)
            safe_error = jnp.maximum(err_norm, 1.0e-12)
            growth = self.safety_factor * safe_error ** (-0.5)
            growth = jnp.clip(growth, self.min_step_factor, self.max_step_factor)
            next_dt = jnp.clip(trial_dt * growth, dt_min, dt_max)

            def _accept(_):
                t_new = t_value + trial_dt
                accepted_y = _project_flat_state_if_needed(trial_y, project_flat)
                return (
                    t_new,
                    accepted_y,
                    next_dt,
                    t_new >= (t_final - 1.0e-15),
                    jnp.asarray(False),
                    fail_code,
                    n_accepted + 1,
                ), (
                    accepted_y,
                    t_new,
                    trial_dt,
                    jnp.asarray(True),
                    jnp.asarray(False),
                    fail_code,
                )

            def _reject(_):
                code = jnp.asarray(1, dtype=jnp.int32)
                reduced_dt = jnp.maximum(jnp.minimum(next_dt, dt_value * 0.5), dt_min)
                fail_now = jnp.logical_and(reduced_dt <= dt_min * (1.0 + 1.0e-12), jnp.logical_not(accepted))
                return (
                    t_value,
                    flat_y,
                    reduced_dt,
                    done,
                    fail_now,
                    code,
                    n_accepted,
                ), (
                    flat_y,
                    t_value,
                    jnp.asarray(0.0, dtype=dtype),
                    jnp.asarray(False),
                    fail_now,
                    code,
                )

            return jax.lax.cond(accepted, _accept, _reject, operand=None)

        def step_fn(carry, _):
            t_value, flat_y, dt_value, done, failed, fail_code, n_accepted = carry

            def _skip(_):
                return carry, (
                    flat_y,
                    t_value,
                    jnp.asarray(0.0, dtype=dtype),
                    jnp.asarray(False),
                    failed,
                    fail_code,
                )

            def _run(_):
                return _attempt_step(carry)

            return jax.lax.cond(jnp.logical_or(done, failed), _skip, _run, operand=None)

        save_n = getattr(self, "save_n", None)
        if save_n is not None and save_n > 1:
            save_times = jnp.linspace(t0, t_final, save_n)
            ys_saved = jnp.zeros((save_n, state_dim), dtype=dtype)
            ts_saved = jnp.zeros((save_n,), dtype=dtype)
            dts_saved = jnp.zeros((save_n,), dtype=dtype)
            accepted_mask_saved = jnp.zeros((save_n,), dtype=bool)
            failed_mask_saved = jnp.zeros((save_n,), dtype=bool)
            fail_codes_saved = jnp.zeros((save_n,), dtype=jnp.int32)
            ys_saved = ys_saved.at[0].set(flat_state0)
            ts_saved = ts_saved.at[0].set(t0)
            accepted_mask_saved = accepted_mask_saved.at[0].set(True)

            def cond_fun(loop_carry):
                t_value, _, _, done, failed, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, step_idx, *_ = loop_carry
                active = jnp.logical_not(jnp.logical_or(done, failed))
                return jnp.logical_and(step_idx < max_total_steps, active)

            def body_fun(loop_carry):
                (
                    t_value,
                    flat_y,
                    dt_value,
                    done,
                    failed,
                    fail_code,
                    n_accepted,
                    step_idx,
                    save_idx,
                    ys,
                    ts,
                    dts,
                    accs,
                    fails,
                    codes,
                ) = loop_carry
                (t_new, y_new, dt_next, done_new, failed_new, fail_code_new, n_acc_new), step_info = step_fn(
                    (t_value, flat_y, dt_value, done, failed, fail_code, n_accepted),
                    None,
                )
                y_out, t_out, dt_out, acc_out, fail_out, code_out = step_info
                save_idx, ys, ts, dts, accs, fails, codes = _fill_saved_slots(
                    save_idx, save_times, t_out, y_out, dt_out, acc_out, fail_out, code_out,
                    ys, ts, dts, accs, fails, codes,
                )
                return (
                    t_new,
                    y_new,
                    dt_next,
                    done_new,
                    failed_new,
                    fail_code_new,
                    n_acc_new,
                    step_idx + 1,
                    save_idx,
                    ys,
                    ts,
                    dts,
                    accs,
                    fails,
                    codes,
                )

            loop_carry = (
                t0,
                flat_state0,
                base_dt,
                jnp.asarray(False),
                jnp.asarray(False),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(1, dtype=jnp.int32),
                ys_saved,
                ts_saved,
                dts_saved,
                accepted_mask_saved,
                failed_mask_saved,
                fail_codes_saved,
            )
            (
                t_f,
                y_f_flat,
                dt_last,
                done_f,
                failed_f,
                fail_code_f,
                n_acc_f,
                _,
                _save_idx_f,
                ys_saved,
                ts_saved,
                dts_saved,
                accepted_mask_saved,
                failed_mask_saved,
                fail_codes_saved,
            ) = jax.lax.while_loop(cond_fun, body_fun, loop_carry)
            return _finalize_custom_solver_output(
                ys_saved, ts_saved, dts_saved, accepted_mask_saved, failed_mask_saved, fail_codes_saved,
                y_f_flat, t_f, done_f, failed_f, fail_code_f, n_acc_f, unpack_flat, state, species,
                temperature_active_mask=temperature_active_mask,
                fixed_temperature_profile=fixed_temperature_profile,
            )

        def cond_fun(loop_carry):
            t_value, _, _, done, failed, _, _, step_idx = loop_carry
            active = jnp.logical_not(jnp.logical_or(done, failed))
            return jnp.logical_and(step_idx < max_total_steps, active)

        def body_fun(loop_carry):
            t_value, flat_y, dt_value, done, failed, fail_code, n_accepted, step_idx = loop_carry
            (t_new, y_new, dt_next, done_new, failed_new, fail_code_new, n_acc_new), _ = step_fn(
                (t_value, flat_y, dt_value, done, failed, fail_code, n_accepted),
                None,
            )
            return t_new, y_new, dt_next, done_new, failed_new, fail_code_new, n_acc_new, step_idx + 1

        loop_carry = (
            t0,
            flat_state0,
            base_dt,
            jnp.asarray(False),
            jnp.asarray(False),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )
        t_f, y_f_flat, dt_last, done_f, failed_f, fail_code_f, n_acc_f, _ = jax.lax.while_loop(cond_fun, body_fun, loop_carry)
        ys_saved_flat = jnp.expand_dims(y_f_flat, axis=0)
        ts_saved = jnp.expand_dims(t_f, axis=0)
        dts_saved = jnp.expand_dims(dt_last, axis=0)
        accepted_mask_saved = jnp.expand_dims(jnp.logical_not(failed_f), axis=0)
        failed_mask_saved = jnp.expand_dims(failed_f, axis=0)
        fail_codes_saved = jnp.expand_dims(fail_code_f, axis=0)
        return _finalize_custom_solver_output(
            ys_saved_flat, ts_saved, dts_saved, accepted_mask_saved, failed_mask_saved, fail_codes_saved,
            y_f_flat, t_f, done_f, failed_f, fail_code_f, n_acc_f, unpack_flat, state, species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class RADAUSolver(TransportSolver):
    t0: float
    t1: float
    dt: float
    rtol: float = 1.0e-6
    atol: float = 1.0e-8
    max_step: float = 1.0
    min_step: float = 1.0e-14
    tol: float = 1.0e-8
    maxiter: int = 20
    linear_solver: str = "gmres"
    gmres_tol: float = 1.0e-8
    gmres_maxiter: int = 80
    error_estimator: str = "embedded2"
    newton_strategy: str = "simplified"
    safety_factor: float = 0.9
    min_step_factor: float = 0.1
    max_step_factor: float = 5.0
    jacobian_reuse_rtol: float = 0.1
    max_jacobian_age: int = 8
    n_steps: int = 0

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
        linear_solver: str = "gmres",
        gmres_tol: float = 1.0e-8,
        gmres_maxiter: int = 80,
        error_estimator: str = "embedded2",
        newton_strategy: str = "simplified",
        safety_factor: float = 0.9,
        min_step_factor: float = 0.1,
        max_step_factor: float = 5.0,
        jacobian_reuse_rtol: float = 0.1,
        max_jacobian_age: int = 8,
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
        object.__setattr__(self, "linear_solver", str(linear_solver).strip().lower())
        object.__setattr__(self, "gmres_tol", float(gmres_tol))
        object.__setattr__(self, "gmres_maxiter", int(max(1, gmres_maxiter)))
        object.__setattr__(self, "error_estimator", str(error_estimator).strip().lower())
        object.__setattr__(self, "newton_strategy", str(newton_strategy).strip().lower())
        object.__setattr__(self, "safety_factor", float(safety_factor))
        object.__setattr__(self, "min_step_factor", float(min_step_factor))
        object.__setattr__(self, "max_step_factor", float(max_step_factor))
        object.__setattr__(self, "jacobian_reuse_rtol", float(jacobian_reuse_rtol))
        object.__setattr__(self, "max_jacobian_age", int(max(0, max_jacobian_age)))
        object.__setattr__(self, "n_steps", n_steps)
        object.__setattr__(self, "save_n", save_n)

    def solve(self, state, vector_field: Callable, *args, **kwargs):
        species = _extract_species_from_args(args)
        temperature_active_mask, fixed_temperature_profile = _extract_fixed_temperature_projection(vector_field)
        state = _project_state_to_quasi_neutrality(
            state,
            species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
        )
        flat_state0, unpack_flat, _unpack_packed, _pack_state, project_flat = _make_solver_state_transform(
            state,
            species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
        )
        dtype = flat_state0.dtype
        sqrt6 = jnp.sqrt(jnp.asarray(6.0, dtype=dtype))
        c = jnp.asarray([(4.0 - sqrt6) / 10.0, (4.0 + sqrt6) / 10.0, 1.0], dtype=dtype)
        a = jnp.asarray(
            [
                [(88.0 - 7.0 * sqrt6) / 360.0, (296.0 - 169.0 * sqrt6) / 1800.0, (-2.0 + 3.0 * sqrt6) / 225.0],
                [(296.0 + 169.0 * sqrt6) / 1800.0, (88.0 + 7.0 * sqrt6) / 360.0, (-2.0 - 3.0 * sqrt6) / 225.0],
                [(16.0 - sqrt6) / 36.0, (16.0 + sqrt6) / 36.0, 1.0 / 9.0],
            ],
            dtype=dtype,
        )
        b = a[2]
        b_embedded = jnp.asarray([0.5 - 0.5 / sqrt6, 0.5 + 0.5 / sqrt6, 0.0], dtype=dtype)
        b_error = b - b_embedded
        t0 = jnp.asarray(self.t0, dtype=dtype)
        t_final = jnp.asarray(self.t1, dtype=dtype)
        dt_min = jnp.asarray(self.min_step, dtype=dtype)
        dt_max = jnp.asarray(self.max_step, dtype=dtype)
        base_dt = jnp.clip(jnp.asarray(self.dt, dtype=dtype), dt_min, dt_max)
        max_total_steps = int(max(32, self.n_steps * 128))
        state_dim = flat_state0.shape[0]
        flat_rhs = _flat_rhs_factory(unpack_flat, vector_field, args, kwargs, project_flat=project_flat)
        error_order = 2.0 if self.error_estimator == "embedded2" else 5.0
        controller_alpha = 0.7 / (error_order + 1.0)
        controller_beta = 0.4 / (error_order + 1.0)
        radau_transform = jnp.asarray(_RADAU3_TRANSFORM, dtype=dtype)
        radau_inv_transform = jnp.asarray(_RADAU3_INV_TRANSFORM, dtype=dtype)
        radau_real_eig = jnp.asarray(_RADAU3_REAL_EIG, dtype=dtype)
        radau_complex_block = jnp.asarray(_RADAU3_COMPLEX_BLOCK, dtype=dtype)
        real_lu0 = jnp.eye(state_dim, dtype=dtype)
        real_piv0 = jnp.arange(state_dim, dtype=jnp.int32)
        complex_dim = 2 * state_dim
        complex_lu0 = jnp.eye(complex_dim, dtype=dtype)
        complex_piv0 = jnp.arange(complex_dim, dtype=jnp.int32)
        use_simplified_newton = self.newton_strategy == "simplified"
        preferred_linear_solver_mode = "direct" if _prefer_dense_direct(self.linear_solver, 2 * state_dim) else self.linear_solver
        use_ntss_direct_path = use_simplified_newton and preferred_linear_solver_mode != "gmres"

        def _error_norm(err_vec, flat_ref, flat_candidate):
            scale = self.atol + self.rtol * jnp.maximum(jnp.abs(flat_ref), jnp.abs(flat_candidate))
            normalized = err_vec / scale
            return jnp.sqrt(jnp.mean(normalized * normalized) + 1.0e-30)

        def _radau_stage_matvec(v_flat, stage_linears, h_value):
            v_stages = v_flat.reshape((3, state_dim))
            coupled = h_value * (a @ v_stages)
            out = tuple(v_stages[i] - stage_linears[i](coupled[i]) for i in range(3))
            return jnp.stack(out, axis=0).reshape((-1,))

        def _transform_stage_stack(stage_stack):
            return radau_inv_transform @ stage_stack

        def _inverse_transform_stage_stack(stage_stack):
            return radau_transform @ stage_stack

        def _make_transformed_radau_stage_solver(
            jacobian_ref,
            h_value,
            linear_solver_mode,
            real_lu_cache,
            real_piv_cache,
            complex_lu_cache,
            complex_piv_cache,
            factor_cache_valid,
            factor_cache_dt,
        ):
            identity_n = jnp.eye(state_dim, dtype=dtype)
            block00 = radau_complex_block[0, 0]
            block01 = radau_complex_block[0, 1]
            block10 = radau_complex_block[1, 0]
            block11 = radau_complex_block[1, 1]
            use_direct_blocks = True if use_ntss_direct_path else (linear_solver_mode != "gmres")
            factor_dt_scale = jnp.maximum(jnp.abs(factor_cache_dt), jnp.asarray(1.0e-14, dtype=dtype))
            factor_dt_close = jnp.abs(h_value - factor_cache_dt) <= self.jacobian_reuse_rtol * factor_dt_scale
            reuse_factorization = jnp.logical_and(factor_cache_valid, jnp.logical_and(use_direct_blocks, factor_dt_close))

            real_matrix = identity_n - h_value * radau_real_eig * jacobian_ref

            def real_matvec(v):
                return v - h_value * radau_real_eig * (jacobian_ref @ v)

            complex_dense = None
            if use_direct_blocks:
                complex_dense = jnp.block(
                    [
                        [identity_n - h_value * block00 * jacobian_ref, -h_value * block01 * jacobian_ref],
                        [-h_value * block10 * jacobian_ref, identity_n - h_value * block11 * jacobian_ref],
                    ]
                )

            def _reuse_factorization(_):
                return real_lu_cache, real_piv_cache, complex_lu_cache, complex_piv_cache

            def _recompute_factorization(_):
                real_lu, real_piv = jax.scipy.linalg.lu_factor(real_matrix)
                complex_lu, complex_piv = jax.scipy.linalg.lu_factor(complex_dense)
                return real_lu, real_piv, complex_lu, complex_piv

            if use_direct_blocks:
                real_lu, real_piv, complex_lu, complex_piv = jax.lax.cond(
                    reuse_factorization,
                    _reuse_factorization,
                    _recompute_factorization,
                    operand=None,
                )
            else:
                real_lu, real_piv, complex_lu, complex_piv = (
                    real_lu_cache,
                    real_piv_cache,
                    complex_lu_cache,
                    complex_piv_cache,
                )

            def complex_matvec(v):
                v1, v2 = jnp.split(v, 2)
                jv1 = jacobian_ref @ v1
                jv2 = jacobian_ref @ v2
                out1 = v1 - h_value * (block00 * jv1 + block01 * jv2)
                out2 = v2 - h_value * (block10 * jv1 + block11 * jv2)
                return jnp.concatenate([out1, out2], axis=0)

            def stage_solver(rhs):
                rhs_stages = rhs.reshape((3, state_dim))
                rhs_transformed = _transform_stage_stack(rhs_stages)
                rhs_real = rhs_transformed[0]
                rhs_complex = rhs_transformed[1:].reshape((-1,))
                if use_direct_blocks:
                    delta_real = jax.scipy.linalg.lu_solve((real_lu, real_piv), rhs_real)
                    delta_complex = jax.scipy.linalg.lu_solve((complex_lu, complex_piv), rhs_complex)
                else:
                    delta_real = _solve_linear_system(
                        matvec=real_matvec,
                        rhs=rhs_real,
                        linear_solver=linear_solver_mode,
                        dense_matrix_builder=None,
                        gmres_tol=self.gmres_tol,
                        gmres_maxiter=self.gmres_maxiter,
                    )
                    delta_complex = _solve_linear_system(
                        matvec=complex_matvec,
                        rhs=rhs_complex,
                        linear_solver=linear_solver_mode,
                        dense_matrix_builder=None,
                        gmres_tol=self.gmres_tol,
                        gmres_maxiter=self.gmres_maxiter,
                    )
                delta_transformed = jnp.concatenate(
                    [delta_real[None, :], delta_complex.reshape((2, state_dim))],
                    axis=0,
                )
                return _inverse_transform_stage_stack(delta_transformed).reshape((-1,))

            factor_cache_valid_out = jnp.asarray(use_direct_blocks)
            factor_cache_dt_out = jnp.where(use_direct_blocks, h_value, factor_cache_dt)
            return (
                stage_solver,
                real_lu,
                real_piv,
                complex_lu,
                complex_piv,
                factor_cache_valid_out,
                factor_cache_dt_out,
            )

        def _single_step(
            flat_y,
            t_value,
            h_value,
            prev_stages,
            prev_dt,
            jacobian_cache,
            cache_valid,
            cache_dt,
            cache_age,
            real_lu_cache,
            real_piv_cache,
            complex_lu_cache,
            complex_piv_cache,
            factor_cache_valid,
            factor_cache_dt,
        ):
            f0 = flat_rhs(t_value, flat_y)
            fallback_guess = jnp.tile(f0, 3)
            use_predictor = jnp.logical_and(
                prev_dt > 0.0,
                jnp.all(jnp.isfinite(prev_stages)),
            )
            z0 = jnp.where(use_predictor, prev_stages * (h_value / prev_dt), fallback_guess)

            def residual_with_stages(z_flat):
                stages = z_flat.reshape((3, state_dim))
                stage_states = flat_y[None, :] + h_value * (a @ stages)
                evals = jnp.stack(
                    [flat_rhs(t_value + c[i] * h_value, stage_states[i]) for i in range(3)],
                    axis=0,
                )
                return (stages - evals).reshape((-1,)), stage_states

            def residual(z_flat):
                residual_flat, _ = residual_with_stages(z_flat)
                return residual_flat

            jacobian_dt_scale = jnp.maximum(jnp.abs(cache_dt), jnp.asarray(1.0e-14, dtype=dtype))
            dt_close = jnp.abs(h_value - cache_dt) <= self.jacobian_reuse_rtol * jacobian_dt_scale
            use_cached_jacobian = jnp.logical_and(
                cache_valid,
                jnp.logical_and(cache_age <= self.max_jacobian_age, dt_close),
            )

            if use_simplified_newton:
                linear_solver_mode = preferred_linear_solver_mode
                def _reuse_jac(_):
                    return jacobian_cache

                def _recompute_jac(_):
                    return jax.jacfwd(lambda y: flat_rhs(t_value, y))(flat_y)

                jacobian_ref = jax.lax.cond(
                    use_cached_jacobian,
                    _reuse_jac,
                    _recompute_jac,
                    operand=None,
                )
                (
                    stage_solver,
                    real_lu_out,
                    real_piv_out,
                    complex_lu_out,
                    complex_piv_out,
                    factor_cache_valid_out,
                    factor_cache_dt_out,
                ) = _make_transformed_radau_stage_solver(
                    jacobian_ref,
                    h_value,
                    linear_solver_mode,
                    real_lu_cache,
                    real_piv_cache,
                    complex_lu_cache,
                    complex_piv_cache,
                    factor_cache_valid,
                    factor_cache_dt,
                )
            else:
                stage_solver = None
                real_lu_out, real_piv_out = real_lu_cache, real_piv_cache
                complex_lu_out, complex_piv_out = complex_lu_cache, complex_piv_cache
                factor_cache_valid_out, factor_cache_dt_out = factor_cache_valid, factor_cache_dt

            def body_fn(newton_state):
                iter_idx, z_cur, delta_norm, residual_norm, prev_residual_norm, theta_est, diverged = newton_state
                residual_cur, stage_states = residual_with_stages(z_cur)
                if use_simplified_newton:
                    delta = stage_solver(-residual_cur)
                else:
                    stage_linears = tuple(
                        jax.linearize(
                            lambda y_stage, t_stage=t_value + c[i] * h_value: flat_rhs(t_stage, y_stage),
                            stage_states[i],
                        )[1]
                        for i in range(3)
                    )
                    dense_builder = None
                    if self.linear_solver != "gmres":
                        basis = jnp.eye(3 * state_dim, dtype=dtype)
                        dense_builder = lambda: jax.vmap(
                            lambda col: _radau_stage_matvec(col, stage_linears, h_value),
                            in_axes=1,
                            out_axes=1,
                        )(basis)
                    delta = _solve_linear_system(
                        matvec=lambda v: _radau_stage_matvec(v, stage_linears, h_value),
                        rhs=-residual_cur,
                        linear_solver=self.linear_solver,
                        dense_matrix_builder=dense_builder,
                        gmres_tol=self.gmres_tol,
                        gmres_maxiter=self.gmres_maxiter,
                    )
                delta = jnp.where(jnp.all(jnp.isfinite(delta)), delta, jnp.zeros_like(delta))
                z_next = z_cur + delta
                current_residual_norm = jnp.linalg.norm(residual_cur)
                safe_prev_residual = jnp.maximum(prev_residual_norm, jnp.asarray(1.0e-30, dtype=dtype))
                theta_raw = current_residual_norm / safe_prev_residual
                theta_candidate = jnp.where(iter_idx > 0, theta_raw, jnp.asarray(0.0, dtype=dtype))
                theta_next = jnp.where(iter_idx > 0, jnp.maximum(theta_est, theta_candidate), theta_est)
                slow_contraction = jnp.logical_and(iter_idx >= 1, theta_candidate > jnp.asarray(0.95, dtype=dtype))
                residual_blowup = jnp.logical_and(iter_idx >= 1, current_residual_norm > prev_residual_norm * jnp.asarray(2.0, dtype=dtype))
                nonfinite_state = jnp.logical_not(jnp.logical_and(jnp.all(jnp.isfinite(delta)), jnp.isfinite(current_residual_norm)))
                diverged_next = jnp.logical_or(diverged, jnp.logical_or(slow_contraction, jnp.logical_or(residual_blowup, nonfinite_state)))
                return (
                    iter_idx + 1,
                    z_next,
                    jnp.linalg.norm(delta),
                    current_residual_norm,
                    current_residual_norm,
                    theta_next,
                    diverged_next,
                )

            def cond_fn(newton_state):
                iter_idx, _, delta_norm, residual_norm, _, _, diverged = newton_state
                active = jnp.logical_or(residual_norm > self.tol, delta_norm > self.tol)
                return jnp.logical_and(jnp.logical_and(iter_idx < self.maxiter, active), jnp.logical_not(diverged))

            init_newton = (
                jnp.asarray(0, dtype=jnp.int32),
                z0,
                jnp.asarray(jnp.inf, dtype=dtype),
                jnp.asarray(jnp.inf, dtype=dtype),
                jnp.asarray(jnp.inf, dtype=dtype),
                jnp.asarray(0.0, dtype=dtype),
                jnp.asarray(False),
            )
            _, z_final, _, _, _, theta_final, diverged_final = jax.lax.while_loop(cond_fn, body_fn, init_newton)
            stages_final = z_final.reshape((3, state_dim))
            final_residual = residual(z_final)
            converged = jnp.logical_and(
                jnp.logical_and(jnp.all(jnp.isfinite(z_final)), jnp.linalg.norm(final_residual) <= self.tol),
                jnp.logical_not(diverged_final),
            )
            flat_next = flat_y + h_value * (b @ stages_final)
            err_vec = h_value * (b_error @ stages_final)
            err_norm = _error_norm(err_vec, flat_y, flat_next)
            theta_safe = jnp.clip(theta_final, jnp.asarray(0.1, dtype=dtype), jnp.asarray(1.5, dtype=dtype))
            newton_shrink = jnp.where(
                converged,
                jnp.asarray(1.0, dtype=dtype),
                jnp.clip(jnp.asarray(0.8, dtype=dtype) / theta_safe, jnp.asarray(0.1, dtype=dtype), jnp.asarray(0.5, dtype=dtype)),
            )
            if use_simplified_newton:
                jacobian_out = jacobian_ref
                cache_valid_out = jnp.asarray(True)
                cache_dt_out = h_value
                cache_age_out = jnp.where(use_cached_jacobian, cache_age + 1, jnp.asarray(0, dtype=jnp.int32))
            else:
                jacobian_out = jacobian_cache
                cache_valid_out = cache_valid
                cache_dt_out = cache_dt
                cache_age_out = cache_age
            return (
                flat_next,
                err_norm,
                converged,
                z_final,
                theta_final,
                jacobian_out,
                cache_valid_out,
                cache_dt_out,
                cache_age_out,
                real_lu_out,
                real_piv_out,
                complex_lu_out,
                complex_piv_out,
                factor_cache_valid_out,
                factor_cache_dt_out,
                newton_shrink,
            )

        def _attempt_step(carry):
            (
                t_value, flat_y, dt_value, done, failed, fail_code, n_accepted, prev_error, prev_stages, prev_dt,
                jacobian_cache, cache_valid, cache_dt, cache_age,
                real_lu_cache, real_piv_cache, complex_lu_cache, complex_piv_cache, factor_cache_valid, factor_cache_dt,
                rejected_last, reject_streak,
            ) = carry
            trial_dt = jnp.minimum(dt_value, t_final - t_value)
            (
                trial_y, err_norm, converged, stage_history, theta_final,
                jacobian_out, cache_valid_out, cache_dt_out, cache_age_out,
                real_lu_out, real_piv_out, complex_lu_out, complex_piv_out, factor_cache_valid_out, factor_cache_dt_out,
                newton_shrink,
            ) = _single_step(
                flat_y, t_value, trial_dt, prev_stages, prev_dt,
                jacobian_cache, cache_valid, cache_dt, cache_age,
                real_lu_cache, real_piv_cache, complex_lu_cache, complex_piv_cache, factor_cache_valid, factor_cache_dt,
            )
            accepted = jnp.logical_and(converged, err_norm <= 1.0)
            safe_error = jnp.maximum(err_norm, 1.0e-12)
            safe_prev_error = jnp.maximum(prev_error, 1.0e-12)
            growth = self.safety_factor * safe_error ** (-controller_alpha) * safe_prev_error ** controller_beta
            growth = jnp.clip(growth, self.min_step_factor, self.max_step_factor)
            max_growth = jnp.asarray(self.max_step_factor, dtype=dtype)
            post_reject_growth_cap = jnp.where(
                reject_streak > 0,
                jnp.asarray(1.2, dtype=dtype),
                max_growth,
            )
            theta_growth_cap = jnp.where(
                theta_final > jnp.asarray(0.9, dtype=dtype),
                jnp.asarray(1.1, dtype=dtype),
                jnp.where(
                    theta_final > jnp.asarray(0.5, dtype=dtype),
                    jnp.asarray(2.0, dtype=dtype),
                    max_growth,
                ),
            )
            growth = jnp.minimum(growth, jnp.minimum(post_reject_growth_cap, theta_growth_cap))
            next_dt = jnp.clip(trial_dt * growth, dt_min, dt_max)

            def _accept(_):
                t_new = t_value + trial_dt
                accepted_y = _project_flat_state_if_needed(trial_y, project_flat)
                return (
                    t_new,
                    accepted_y,
                    next_dt,
                    t_new >= (t_final - 1.0e-15),
                    jnp.asarray(False),
                    fail_code,
                    n_accepted + 1,
                    safe_error,
                    stage_history,
                    trial_dt,
                    jacobian_out,
                    cache_valid_out,
                    cache_dt_out,
                    cache_age_out,
                    real_lu_out,
                    real_piv_out,
                    complex_lu_out,
                    complex_piv_out,
                    factor_cache_valid_out,
                    factor_cache_dt_out,
                    jnp.asarray(False),
                    jnp.asarray(0, dtype=jnp.int32),
                ), (
                    accepted_y,
                    t_new,
                    trial_dt,
                    jnp.asarray(True),
                    jnp.asarray(False),
                    fail_code,
                )

            def _reject(_):
                code = jnp.where(
                    converged,
                    jnp.asarray(2, dtype=jnp.int32),
                    jnp.asarray(1, dtype=jnp.int32),
                )
                reject_streak_next = jnp.minimum(reject_streak + 1, jnp.asarray(8, dtype=jnp.int32))
                rejection_target = jnp.where(converged, next_dt, trial_dt * newton_shrink)
                rejection_cap = jnp.where(
                    reject_streak > 0,
                    jnp.asarray(0.25, dtype=dtype),
                    jnp.asarray(0.5, dtype=dtype),
                )
                reduced_dt = jnp.maximum(
                    jnp.minimum(rejection_target, trial_dt * rejection_cap),
                    dt_min,
                )
                fail_now = jnp.logical_and(reduced_dt <= dt_min * (1.0 + 1.0e-12), jnp.logical_not(accepted))
                return (
                    t_value,
                    flat_y,
                    reduced_dt,
                    done,
                    fail_now,
                    jnp.where(fail_now, code, jnp.asarray(0, dtype=jnp.int32)),
                    n_accepted,
                    prev_error,
                    prev_stages,
                    prev_dt,
                    jacobian_out,
                    cache_valid_out,
                    cache_dt_out,
                    cache_age_out,
                    real_lu_out,
                    real_piv_out,
                    complex_lu_out,
                    complex_piv_out,
                    factor_cache_valid_out,
                    factor_cache_dt_out,
                    jnp.asarray(True),
                    reject_streak_next,
                ), (
                    flat_y,
                    t_value,
                    jnp.asarray(0.0, dtype=dtype),
                    jnp.asarray(False),
                    fail_now,
                    code,
                )

            return jax.lax.cond(accepted, _accept, _reject, operand=None)

        def step_fn(carry, _):
            (
                t_value, flat_y, dt_value, done, failed, fail_code, n_accepted, prev_error, prev_stages, prev_dt,
                jacobian_cache, cache_valid, cache_dt, cache_age,
                real_lu_cache, real_piv_cache, complex_lu_cache, complex_piv_cache, factor_cache_valid, factor_cache_dt,
                rejected_last, reject_streak,
            ) = carry

            def _skip(_):
                return carry, (
                    flat_y,
                    t_value,
                    jnp.asarray(0.0, dtype=dtype),
                    jnp.asarray(False),
                    failed,
                    fail_code,
                )

            def _run(_):
                return _attempt_step(carry)

            return jax.lax.cond(jnp.logical_or(done, failed), _skip, _run, operand=None)

        save_n = getattr(self, "save_n", None)
        if save_n is not None and save_n > 1:
            save_times = jnp.linspace(t0, t_final, save_n)
            ys_saved = jnp.zeros((save_n, state_dim), dtype=dtype)
            ts_saved = jnp.zeros((save_n,), dtype=dtype)
            dts_saved = jnp.zeros((save_n,), dtype=dtype)
            accepted_mask_saved = jnp.zeros((save_n,), dtype=bool)
            failed_mask_saved = jnp.zeros((save_n,), dtype=bool)
            fail_codes_saved = jnp.zeros((save_n,), dtype=jnp.int32)
            ys_saved = ys_saved.at[0].set(flat_state0)
            ts_saved = ts_saved.at[0].set(t0)
            accepted_mask_saved = accepted_mask_saved.at[0].set(True)

            def cond_fun(loop_carry):
                t_value, _, _, done, failed, _, _, step_idx, *_ = loop_carry
                active = jnp.logical_not(jnp.logical_or(done, failed))
                return jnp.logical_and(step_idx < max_total_steps, active)

            def body_fun(loop_carry):
                (
                    t_value,
                    flat_y,
                    dt_value,
                    done,
                    failed,
                    fail_code,
                    n_accepted,
                    prev_error,
                    prev_stages,
                    prev_dt,
                    jacobian_cache,
                    cache_valid,
                    cache_dt,
                    cache_age,
                    real_lu_cache,
                    real_piv_cache,
                    complex_lu_cache,
                    complex_piv_cache,
                    factor_cache_valid,
                    factor_cache_dt,
                    rejected_last,
                    reject_streak,
                    step_idx,
                    save_idx,
                    ys,
                    ts,
                    dts,
                    accs,
                    fails,
                    codes,
                ) = loop_carry
                (
                    t_new, y_new, dt_next, done_new, failed_new, fail_code_new, n_acc_new,
                    prev_error_new, prev_stages_new, prev_dt_new,
                    jacobian_new, cache_valid_new, cache_dt_new, cache_age_new,
                    real_lu_new, real_piv_new, complex_lu_new, complex_piv_new, factor_cache_valid_new, factor_cache_dt_new,
                    rejected_last_new, reject_streak_new,
                ), step_info = step_fn(
                    (
                        t_value, flat_y, dt_value, done, failed, fail_code, n_accepted,
                        prev_error, prev_stages, prev_dt,
                        jacobian_cache, cache_valid, cache_dt, cache_age,
                        real_lu_cache, real_piv_cache, complex_lu_cache, complex_piv_cache, factor_cache_valid, factor_cache_dt,
                        rejected_last, reject_streak,
                    ),
                    None,
                )
                y_out, t_out, dt_out, acc_out, fail_out, code_out = step_info
                save_idx, ys, ts, dts, accs, fails, codes = _fill_saved_slots(
                    save_idx, save_times, t_out, y_out, dt_out, acc_out, fail_out, code_out,
                    ys, ts, dts, accs, fails, codes,
                )
                return (
                    t_new,
                    y_new,
                    dt_next,
                    done_new,
                    failed_new,
                    fail_code_new,
                    n_acc_new,
                    prev_error_new,
                    prev_stages_new,
                    prev_dt_new,
                    jacobian_new,
                    cache_valid_new,
                    cache_dt_new,
                    cache_age_new,
                    real_lu_new,
                    real_piv_new,
                    complex_lu_new,
                    complex_piv_new,
                    factor_cache_valid_new,
                    factor_cache_dt_new,
                    rejected_last_new,
                    reject_streak_new,
                    step_idx + 1,
                    save_idx,
                    ys,
                    ts,
                    dts,
                    accs,
                    fails,
                    codes,
                )

            loop_carry = (
                t0,
                flat_state0,
                base_dt,
                jnp.asarray(False),
                jnp.asarray(False),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(1.0, dtype=dtype),
                jnp.tile(flat_rhs(t0, flat_state0), 3),
                jnp.asarray(0.0, dtype=dtype),
                jnp.zeros((state_dim, state_dim), dtype=dtype),
                jnp.asarray(False),
                jnp.asarray(0.0, dtype=dtype),
                jnp.asarray(0, dtype=jnp.int32),
                real_lu0,
                real_piv0,
                complex_lu0,
                complex_piv0,
                jnp.asarray(False),
                jnp.asarray(0.0, dtype=dtype),
                jnp.asarray(False),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(1, dtype=jnp.int32),
                ys_saved,
                ts_saved,
                dts_saved,
                accepted_mask_saved,
                failed_mask_saved,
                fail_codes_saved,
            )
            (
                t_f,
                y_f_flat,
                dt_last,
                done_f,
                failed_f,
                fail_code_f,
                n_acc_f,
                _prev_err_f,
                _prev_stages_f,
                _prev_dt_f,
                _jacobian_f,
                _cache_valid_f,
                _cache_dt_f,
                _cache_age_f,
                _real_lu_f,
                _real_piv_f,
                _complex_lu_f,
                _complex_piv_f,
                _factor_cache_valid_f,
                _factor_cache_dt_f,
                _rejected_last_f,
                _reject_streak_f,
                _,
                _save_idx_f,
                ys_saved,
                ts_saved,
                dts_saved,
                accepted_mask_saved,
                failed_mask_saved,
                fail_codes_saved,
            ) = jax.lax.while_loop(cond_fun, body_fun, loop_carry)
            return _finalize_custom_solver_output(
                ys_saved, ts_saved, dts_saved, accepted_mask_saved, failed_mask_saved, fail_codes_saved,
                y_f_flat, t_f, done_f, failed_f, fail_code_f, n_acc_f, unpack_flat, state, species,
                temperature_active_mask=temperature_active_mask,
                fixed_temperature_profile=fixed_temperature_profile,
            )

        def cond_fun(loop_carry):
            t_value, _, _, done, failed, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, step_idx = loop_carry
            active = jnp.logical_not(jnp.logical_or(done, failed))
            return jnp.logical_and(step_idx < max_total_steps, active)

        def body_fun(loop_carry):
            (
                t_value, flat_y, dt_value, done, failed, fail_code, n_accepted, prev_error, prev_stages, prev_dt,
                jacobian_cache, cache_valid, cache_dt, cache_age,
                real_lu_cache, real_piv_cache, complex_lu_cache, complex_piv_cache, factor_cache_valid, factor_cache_dt,
                rejected_last, reject_streak,
                step_idx,
            ) = loop_carry
            (
                t_new, y_new, dt_next, done_new, failed_new, fail_code_new, n_acc_new,
                prev_error_new, prev_stages_new, prev_dt_new,
                jacobian_new, cache_valid_new, cache_dt_new, cache_age_new,
                real_lu_new, real_piv_new, complex_lu_new, complex_piv_new, factor_cache_valid_new, factor_cache_dt_new,
                rejected_last_new, reject_streak_new,
            ), _ = step_fn(
                (
                    t_value, flat_y, dt_value, done, failed, fail_code, n_accepted,
                    prev_error, prev_stages, prev_dt,
                    jacobian_cache, cache_valid, cache_dt, cache_age,
                    real_lu_cache, real_piv_cache, complex_lu_cache, complex_piv_cache, factor_cache_valid, factor_cache_dt,
                    rejected_last, reject_streak,
                ),
                None,
            )
            return (
                t_new,
                y_new,
                dt_next,
                done_new,
                failed_new,
                fail_code_new,
                n_acc_new,
                prev_error_new,
                prev_stages_new,
                prev_dt_new,
                jacobian_new,
                cache_valid_new,
                cache_dt_new,
                cache_age_new,
                real_lu_new,
                real_piv_new,
                complex_lu_new,
                complex_piv_new,
                factor_cache_valid_new,
                factor_cache_dt_new,
                rejected_last_new,
                reject_streak_new,
                step_idx + 1,
            )

        loop_carry = (
            t0,
            flat_state0,
            base_dt,
            jnp.asarray(False),
            jnp.asarray(False),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(1.0, dtype=dtype),
            jnp.tile(flat_rhs(t0, flat_state0), 3),
            jnp.asarray(0.0, dtype=dtype),
            jnp.zeros((state_dim, state_dim), dtype=dtype),
            jnp.asarray(False),
            jnp.asarray(0.0, dtype=dtype),
            jnp.asarray(0, dtype=jnp.int32),
            real_lu0,
            real_piv0,
            complex_lu0,
            complex_piv0,
            jnp.asarray(False),
            jnp.asarray(0.0, dtype=dtype),
            jnp.asarray(False),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )
        (
            t_f, y_f_flat, dt_last, done_f, failed_f, fail_code_f, n_acc_f,
            _prev_err_f, _prev_stages_f, _prev_dt_f,
            _jacobian_f, _cache_valid_f, _cache_dt_f, _cache_age_f,
            _real_lu_f, _real_piv_f, _complex_lu_f, _complex_piv_f, _factor_cache_valid_f, _factor_cache_dt_f,
            _rejected_last_f, _reject_streak_f,
            _
        ) = jax.lax.while_loop(cond_fun, body_fun, loop_carry)
        ys_saved_flat = jnp.expand_dims(y_f_flat, axis=0)
        ts_saved = jnp.expand_dims(t_f, axis=0)
        dts_saved = jnp.expand_dims(dt_last, axis=0)
        accepted_mask_saved = jnp.expand_dims(jnp.logical_not(failed_f), axis=0)
        failed_mask_saved = jnp.expand_dims(failed_f, axis=0)
        fail_codes_saved = jnp.expand_dims(fail_code_f, axis=0)
        return _finalize_custom_solver_output(
            ys_saved_flat, ts_saved, dts_saved, accepted_mask_saved, failed_mask_saved, fail_codes_saved,
            y_f_flat, t_f, done_f, failed_f, fail_code_f, n_acc_f, unpack_flat, state, species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
        )

    def _solve_standard(self, state, vector_field: Callable, *args, **kwargs):
        species = _extract_species_from_args(args)
        temperature_active_mask, fixed_temperature_profile = _extract_fixed_temperature_projection(vector_field)
        state = _project_state_to_quasi_neutrality(
            state,
            species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
        )
        flat_state0, unpack_flat, _unpack_packed, _pack_state, project_flat = _make_solver_state_transform(
            state,
            species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
        )
        dtype = flat_state0.dtype
        sqrt6 = jnp.sqrt(jnp.asarray(6.0, dtype=dtype))
        c = jnp.asarray([(4.0 - sqrt6) / 10.0, (4.0 + sqrt6) / 10.0, 1.0], dtype=dtype)
        a = jnp.asarray(
            [
                [(88.0 - 7.0 * sqrt6) / 360.0, (296.0 - 169.0 * sqrt6) / 1800.0, (-2.0 + 3.0 * sqrt6) / 225.0],
                [(296.0 + 169.0 * sqrt6) / 1800.0, (88.0 + 7.0 * sqrt6) / 360.0, (-2.0 - 3.0 * sqrt6) / 225.0],
                [(16.0 - sqrt6) / 36.0, (16.0 + sqrt6) / 36.0, 1.0 / 9.0],
            ],
            dtype=dtype,
        )
        b = a[2]
        b_embedded = jnp.asarray([0.5 - 0.5 / sqrt6, 0.5 + 0.5 / sqrt6, 0.0], dtype=dtype)
        b_error = b - b_embedded
        t0 = jnp.asarray(self.t0, dtype=dtype)
        t_final = jnp.asarray(self.t1, dtype=dtype)
        dt_min = jnp.asarray(self.min_step, dtype=dtype)
        dt_max = jnp.asarray(self.max_step, dtype=dtype)
        base_dt = jnp.clip(jnp.asarray(self.dt, dtype=dtype), dt_min, dt_max)
        max_total_steps = int(max(32, self.n_steps * 128))
        state_dim = flat_state0.shape[0]
        flat_rhs = _flat_rhs_factory(unpack_flat, vector_field, args, kwargs, project_flat=project_flat)
        error_order = 2.0 if self.error_estimator == "embedded2" else 5.0
        controller_alpha = 0.7 / (error_order + 1.0)
        controller_beta = 0.4 / (error_order + 1.0)
        radau_transform = jnp.asarray(_RADAU3_TRANSFORM, dtype=dtype)
        radau_inv_transform = jnp.asarray(_RADAU3_INV_TRANSFORM, dtype=dtype)
        radau_real_eig = jnp.asarray(_RADAU3_REAL_EIG, dtype=dtype)
        radau_complex_block = jnp.asarray(_RADAU3_COMPLEX_BLOCK, dtype=dtype)
        real_lu0 = jnp.eye(state_dim, dtype=dtype)
        real_piv0 = jnp.arange(state_dim, dtype=jnp.int32)
        complex_dim = 2 * state_dim
        complex_lu0 = jnp.eye(complex_dim, dtype=dtype)
        complex_piv0 = jnp.arange(complex_dim, dtype=jnp.int32)

        def _error_norm(err_vec, flat_ref, flat_candidate):
            scale = self.atol + self.rtol * jnp.maximum(jnp.abs(flat_ref), jnp.abs(flat_candidate))
            normalized = err_vec / scale
            return jnp.sqrt(jnp.mean(normalized * normalized) + 1.0e-30)

        def _transform_stage_stack(stage_stack):
            return radau_inv_transform @ stage_stack

        def _inverse_transform_stage_stack(stage_stack):
            return radau_transform @ stage_stack

        def _make_transformed_radau_stage_solver(
            jacobian_ref,
            h_value,
            real_lu_cache,
            real_piv_cache,
            complex_lu_cache,
            complex_piv_cache,
            factor_cache_valid,
            factor_cache_dt,
        ):
            identity_n = jnp.eye(state_dim, dtype=dtype)
            block00 = radau_complex_block[0, 0]
            block01 = radau_complex_block[0, 1]
            block10 = radau_complex_block[1, 0]
            block11 = radau_complex_block[1, 1]
            factor_dt_scale = jnp.maximum(jnp.abs(factor_cache_dt), jnp.asarray(1.0e-14, dtype=dtype))
            factor_dt_close = jnp.abs(h_value - factor_cache_dt) <= self.jacobian_reuse_rtol * factor_dt_scale
            reuse_factorization = jnp.logical_and(factor_cache_valid, factor_dt_close)

            real_matrix = identity_n - h_value * radau_real_eig * jacobian_ref
            complex_dense = jnp.block(
                [
                    [identity_n - h_value * block00 * jacobian_ref, -h_value * block01 * jacobian_ref],
                    [-h_value * block10 * jacobian_ref, identity_n - h_value * block11 * jacobian_ref],
                ]
            )

            def _reuse_factorization(_):
                return real_lu_cache, real_piv_cache, complex_lu_cache, complex_piv_cache

            def _recompute_factorization(_):
                real_lu, real_piv = jax.scipy.linalg.lu_factor(real_matrix)
                complex_lu, complex_piv = jax.scipy.linalg.lu_factor(complex_dense)
                return real_lu, real_piv, complex_lu, complex_piv

            real_lu, real_piv, complex_lu, complex_piv = jax.lax.cond(
                reuse_factorization,
                _reuse_factorization,
                _recompute_factorization,
                operand=None,
            )

            def stage_solver(rhs):
                rhs_stages = rhs.reshape((3, state_dim))
                rhs_transformed = _transform_stage_stack(rhs_stages)
                rhs_real = rhs_transformed[0]
                rhs_complex = rhs_transformed[1:].reshape((-1,))
                delta_real = jax.scipy.linalg.lu_solve((real_lu, real_piv), rhs_real)
                delta_complex = jax.scipy.linalg.lu_solve((complex_lu, complex_piv), rhs_complex)
                delta_transformed = jnp.concatenate(
                    [delta_real[None, :], delta_complex.reshape((2, state_dim))],
                    axis=0,
                )
                return _inverse_transform_stage_stack(delta_transformed).reshape((-1,))

            return (
                stage_solver,
                real_lu,
                real_piv,
                complex_lu,
                complex_piv,
                jnp.asarray(True),
                h_value,
            )

        def _single_step(
            flat_y,
            t_value,
            h_value,
            prev_stages,
            prev_dt,
            jacobian_cache,
            cache_valid,
            cache_dt,
            cache_age,
            real_lu_cache,
            real_piv_cache,
            complex_lu_cache,
            complex_piv_cache,
            factor_cache_valid,
            factor_cache_dt,
        ):
            f0 = flat_rhs(t_value, flat_y)
            fallback_guess = jnp.tile(f0, 3)
            use_predictor = jnp.logical_and(prev_dt > 0.0, jnp.all(jnp.isfinite(prev_stages)))
            z0 = jnp.where(use_predictor, prev_stages * (h_value / prev_dt), fallback_guess)

            def residual_with_stages(z_flat):
                stages = z_flat.reshape((3, state_dim))
                stage_states = flat_y[None, :] + h_value * (a @ stages)
                evals = jnp.stack(
                    [flat_rhs(t_value + c[i] * h_value, stage_states[i]) for i in range(3)],
                    axis=0,
                )
                return (stages - evals).reshape((-1,)), stage_states

            def residual(z_flat):
                residual_flat, _ = residual_with_stages(z_flat)
                return residual_flat

            jacobian_dt_scale = jnp.maximum(jnp.abs(cache_dt), jnp.asarray(1.0e-14, dtype=dtype))
            dt_close = jnp.abs(h_value - cache_dt) <= self.jacobian_reuse_rtol * jacobian_dt_scale
            use_cached_jacobian = jnp.logical_and(
                cache_valid,
                jnp.logical_and(cache_age <= self.max_jacobian_age, dt_close),
            )

            def _reuse_jac(_):
                return jacobian_cache

            def _recompute_jac(_):
                return jax.jacfwd(lambda y: flat_rhs(t_value, y))(flat_y)

            jacobian_ref = jax.lax.cond(
                use_cached_jacobian,
                _reuse_jac,
                _recompute_jac,
                operand=None,
            )
            (
                stage_solver,
                real_lu_out,
                real_piv_out,
                complex_lu_out,
                complex_piv_out,
                factor_cache_valid_out,
                factor_cache_dt_out,
            ) = _make_transformed_radau_stage_solver(
                jacobian_ref,
                h_value,
                real_lu_cache,
                real_piv_cache,
                complex_lu_cache,
                complex_piv_cache,
                factor_cache_valid,
                factor_cache_dt,
            )

            def body_fn(newton_state):
                iter_idx, z_cur, delta_norm, residual_norm, prev_residual_norm, theta_est, diverged = newton_state
                residual_cur, _stage_states = residual_with_stages(z_cur)
                delta = stage_solver(-residual_cur)
                delta = jnp.where(jnp.all(jnp.isfinite(delta)), delta, jnp.zeros_like(delta))
                z_next = z_cur + delta
                current_residual_norm = jnp.linalg.norm(residual_cur)
                safe_prev_residual = jnp.maximum(prev_residual_norm, jnp.asarray(1.0e-30, dtype=dtype))
                theta_raw = current_residual_norm / safe_prev_residual
                theta_candidate = jnp.where(iter_idx > 0, theta_raw, jnp.asarray(0.0, dtype=dtype))
                theta_next = jnp.where(iter_idx > 0, jnp.maximum(theta_est, theta_candidate), theta_est)
                slow_contraction = jnp.logical_and(iter_idx >= 1, theta_candidate > jnp.asarray(0.95, dtype=dtype))
                residual_blowup = jnp.logical_and(
                    iter_idx >= 1,
                    current_residual_norm > prev_residual_norm * jnp.asarray(2.0, dtype=dtype),
                )
                nonfinite_state = jnp.logical_not(
                    jnp.logical_and(jnp.all(jnp.isfinite(delta)), jnp.isfinite(current_residual_norm))
                )
                diverged_next = jnp.logical_or(
                    diverged,
                    jnp.logical_or(slow_contraction, jnp.logical_or(residual_blowup, nonfinite_state)),
                )
                return (
                    iter_idx + 1,
                    z_next,
                    jnp.linalg.norm(delta),
                    current_residual_norm,
                    current_residual_norm,
                    theta_next,
                    diverged_next,
                )

            def cond_fn(newton_state):
                iter_idx, _, delta_norm, residual_norm, _, _, diverged = newton_state
                active = jnp.logical_or(residual_norm > self.tol, delta_norm > self.tol)
                return jnp.logical_and(
                    jnp.logical_and(iter_idx < self.maxiter, active),
                    jnp.logical_not(diverged),
                )

            init_newton = (
                jnp.asarray(0, dtype=jnp.int32),
                z0,
                jnp.asarray(jnp.inf, dtype=dtype),
                jnp.asarray(jnp.inf, dtype=dtype),
                jnp.asarray(jnp.inf, dtype=dtype),
                jnp.asarray(0.0, dtype=dtype),
                jnp.asarray(False),
            )
            _, z_final, _, _, _, theta_final, diverged_final = jax.lax.while_loop(cond_fn, body_fn, init_newton)
            stages_final = z_final.reshape((3, state_dim))
            final_residual = residual(z_final)
            converged = jnp.logical_and(
                jnp.logical_and(jnp.all(jnp.isfinite(z_final)), jnp.linalg.norm(final_residual) <= self.tol),
                jnp.logical_not(diverged_final),
            )
            flat_next = flat_y + h_value * (b @ stages_final)
            err_vec = h_value * (b_error @ stages_final)
            err_norm = _error_norm(err_vec, flat_y, flat_next)
            theta_safe = jnp.clip(theta_final, jnp.asarray(0.1, dtype=dtype), jnp.asarray(1.5, dtype=dtype))
            newton_shrink = jnp.where(
                converged,
                jnp.asarray(1.0, dtype=dtype),
                jnp.clip(jnp.asarray(0.8, dtype=dtype) / theta_safe, jnp.asarray(0.1, dtype=dtype), jnp.asarray(0.5, dtype=dtype)),
            )
            jacobian_out = jacobian_ref
            cache_valid_out = jnp.asarray(True)
            cache_dt_out = h_value
            cache_age_out = jnp.where(use_cached_jacobian, cache_age + 1, jnp.asarray(0, dtype=jnp.int32))
            return (
                flat_next,
                err_norm,
                converged,
                z_final,
                theta_final,
                jacobian_out,
                cache_valid_out,
                cache_dt_out,
                cache_age_out,
                real_lu_out,
                real_piv_out,
                complex_lu_out,
                complex_piv_out,
                factor_cache_valid_out,
                factor_cache_dt_out,
                newton_shrink,
            )

        def _attempt_step(carry):
            (
                t_value, flat_y, dt_value, done, failed, fail_code, n_accepted, prev_error, prev_stages, prev_dt,
                jacobian_cache, cache_valid, cache_dt, cache_age,
                real_lu_cache, real_piv_cache, complex_lu_cache, complex_piv_cache, factor_cache_valid, factor_cache_dt,
                rejected_last, reject_streak,
            ) = carry
            trial_dt = jnp.minimum(dt_value, t_final - t_value)
            (
                trial_y, err_norm, converged, stage_history, theta_final,
                jacobian_out, cache_valid_out, cache_dt_out, cache_age_out,
                real_lu_out, real_piv_out, complex_lu_out, complex_piv_out, factor_cache_valid_out, factor_cache_dt_out,
                newton_shrink,
            ) = _single_step(
                flat_y, t_value, trial_dt, prev_stages, prev_dt,
                jacobian_cache, cache_valid, cache_dt, cache_age,
                real_lu_cache, real_piv_cache, complex_lu_cache, complex_piv_cache, factor_cache_valid, factor_cache_dt,
            )
            accepted = jnp.logical_and(converged, err_norm <= 1.0)
            safe_error = jnp.maximum(err_norm, 1.0e-12)
            safe_prev_error = jnp.maximum(prev_error, 1.0e-12)
            growth = self.safety_factor * safe_error ** (-controller_alpha) * safe_prev_error ** controller_beta
            growth = jnp.clip(growth, self.min_step_factor, self.max_step_factor)
            max_growth = jnp.asarray(self.max_step_factor, dtype=dtype)
            easy_history_boost = jnp.where(
                jnp.logical_and(easy_accept_streak > 0, jnp.logical_not(rejected_last)),
                jnp.asarray(1.15, dtype=dtype),
                jnp.asarray(1.0, dtype=dtype),
            )
            growth = jnp.minimum(growth * easy_history_boost, max_growth)
            post_reject_growth_cap = jnp.where(reject_streak > 0, jnp.asarray(1.2, dtype=dtype), max_growth)
            theta_growth_cap = jnp.where(
                theta_final > jnp.asarray(0.9, dtype=dtype),
                jnp.asarray(1.1, dtype=dtype),
                jnp.where(theta_final > jnp.asarray(0.5, dtype=dtype), jnp.asarray(2.0, dtype=dtype), max_growth),
            )
            easy_growth_floor = jnp.where(
                jnp.logical_and(
                    easy_accept_streak >= 2,
                    jnp.logical_and(
                        safe_error < jnp.asarray(0.08, dtype=dtype),
                        theta_final < jnp.asarray(0.2, dtype=dtype),
                    ),
                ),
                jnp.asarray(1.8, dtype=dtype),
                jnp.asarray(1.0, dtype=dtype),
            )
            growth = jnp.maximum(growth, easy_growth_floor)
            growth = jnp.minimum(growth, jnp.minimum(post_reject_growth_cap, theta_growth_cap))
            next_dt = jnp.clip(trial_dt * growth, dt_min, dt_max)

            def _accept(_):
                t_new = t_value + trial_dt
                accepted_y = _project_flat_state_if_needed(trial_y, project_flat)
                easy_accept = jnp.logical_and(
                    jnp.logical_not(rejected_last),
                    jnp.logical_and(
                        safe_error < jnp.asarray(0.2, dtype=dtype),
                        theta_final < jnp.asarray(0.35, dtype=dtype),
                    ),
                )
                easy_accept_streak_next = jnp.where(
                    easy_accept,
                    jnp.minimum(easy_accept_streak + 1, jnp.asarray(6, dtype=jnp.int32)),
                    jnp.asarray(0, dtype=jnp.int32),
                )
                return (
                    t_new, accepted_y, next_dt, t_new >= (t_final - 1.0e-15), jnp.asarray(False), fail_code,
                    n_accepted + 1, safe_error, stage_history, trial_dt,
                    jacobian_out, cache_valid_out, cache_dt_out, cache_age_out,
                    real_lu_out, real_piv_out, complex_lu_out, complex_piv_out, factor_cache_valid_out, factor_cache_dt_out,
                    jnp.asarray(False), jnp.asarray(0, dtype=jnp.int32), easy_accept_streak_next,
                ), (
                    accepted_y, t_new, trial_dt, jnp.asarray(True), jnp.asarray(False), fail_code,
                )

            def _reject(_):
                code = jnp.where(converged, jnp.asarray(2, dtype=jnp.int32), jnp.asarray(1, dtype=jnp.int32))
                reject_streak_next = jnp.minimum(reject_streak + 1, jnp.asarray(8, dtype=jnp.int32))
                rejection_target = jnp.where(converged, next_dt, trial_dt * newton_shrink)
                rejection_cap = jnp.where(reject_streak > 0, jnp.asarray(0.25, dtype=dtype), jnp.asarray(0.5, dtype=dtype))
                reduced_dt = jnp.maximum(jnp.minimum(rejection_target, trial_dt * rejection_cap), dt_min)
                fail_now = jnp.logical_and(reduced_dt <= dt_min * (1.0 + 1.0e-12), jnp.logical_not(accepted))
                return (
                    t_value, flat_y, reduced_dt, done, fail_now, jnp.where(fail_now, code, jnp.asarray(0, dtype=jnp.int32)),
                    n_accepted, prev_error, prev_stages, prev_dt,
                    jacobian_out, cache_valid_out, cache_dt_out, cache_age_out,
                    real_lu_out, real_piv_out, complex_lu_out, complex_piv_out, factor_cache_valid_out, factor_cache_dt_out,
                    jnp.asarray(True), reject_streak_next, jnp.asarray(0, dtype=jnp.int32),
                ), (
                    flat_y, t_value, jnp.asarray(0.0, dtype=dtype), jnp.asarray(False), fail_now, code,
                )

            return jax.lax.cond(accepted, _accept, _reject, operand=None)

        def step_fn(carry, _):
            (
                t_value, flat_y, dt_value, done, failed, fail_code, n_accepted, prev_error, prev_stages, prev_dt,
                jacobian_cache, cache_valid, cache_dt, cache_age,
                real_lu_cache, real_piv_cache, complex_lu_cache, complex_piv_cache, factor_cache_valid, factor_cache_dt,
                rejected_last, reject_streak, easy_accept_streak,
            ) = carry

            def _skip(_):
                return carry, (flat_y, t_value, jnp.asarray(0.0, dtype=dtype), jnp.asarray(False), failed, fail_code)

            def _run(_):
                return _attempt_step(carry)

            return jax.lax.cond(jnp.logical_or(done, failed), _skip, _run, operand=None)

        save_n = getattr(self, "save_n", None)
        if save_n is not None and save_n > 1:
            save_times = jnp.linspace(t0, t_final, save_n)
            ys_saved = jnp.zeros((save_n, state_dim), dtype=dtype)
            ts_saved = jnp.zeros((save_n,), dtype=dtype)
            dts_saved = jnp.zeros((save_n,), dtype=dtype)
            accepted_mask_saved = jnp.zeros((save_n,), dtype=bool)
            failed_mask_saved = jnp.zeros((save_n,), dtype=bool)
            fail_codes_saved = jnp.zeros((save_n,), dtype=jnp.int32)
            ys_saved = ys_saved.at[0].set(flat_state0)
            ts_saved = ts_saved.at[0].set(t0)
            accepted_mask_saved = accepted_mask_saved.at[0].set(True)

            def cond_fun(loop_carry):
                t_value, _, _, done, failed, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, step_idx, *_ = loop_carry
                active = jnp.logical_not(jnp.logical_or(done, failed))
                return jnp.logical_and(step_idx < max_total_steps, active)

            def body_fun(loop_carry):
                (
                    t_value, flat_y, dt_value, done, failed, fail_code, n_accepted, prev_error, prev_stages, prev_dt,
                    jacobian_cache, cache_valid, cache_dt, cache_age,
                    real_lu_cache, real_piv_cache, complex_lu_cache, complex_piv_cache, factor_cache_valid, factor_cache_dt,
                    rejected_last, reject_streak, easy_accept_streak,
                    step_idx, save_idx, ys, ts, dts, accs, fails, codes,
                ) = loop_carry
                (
                    t_new, y_new, dt_next, done_new, failed_new, fail_code_new, n_acc_new,
                    prev_error_new, prev_stages_new, prev_dt_new,
                    jacobian_new, cache_valid_new, cache_dt_new, cache_age_new,
                    real_lu_new, real_piv_new, complex_lu_new, complex_piv_new, factor_cache_valid_new, factor_cache_dt_new,
                    rejected_last_new, reject_streak_new, easy_accept_streak_new,
                ), step_info = step_fn(
                    (
                        t_value, flat_y, dt_value, done, failed, fail_code, n_accepted,
                        prev_error, prev_stages, prev_dt,
                        jacobian_cache, cache_valid, cache_dt, cache_age,
                        real_lu_cache, real_piv_cache, complex_lu_cache, complex_piv_cache, factor_cache_valid, factor_cache_dt,
                        rejected_last, reject_streak, easy_accept_streak,
                    ),
                    None,
                )
                y_out, t_out, dt_out, acc_out, fail_out, code_out = step_info
                save_idx, ys, ts, dts, accs, fails, codes = _fill_saved_slots(
                    save_idx, save_times, t_out, y_out, dt_out, acc_out, fail_out, code_out,
                    ys, ts, dts, accs, fails, codes,
                )
                return (
                    t_new, y_new, dt_next, done_new, failed_new, fail_code_new, n_acc_new,
                    prev_error_new, prev_stages_new, prev_dt_new,
                    jacobian_new, cache_valid_new, cache_dt_new, cache_age_new,
                    real_lu_new, real_piv_new, complex_lu_new, complex_piv_new, factor_cache_valid_new, factor_cache_dt_new,
                    rejected_last_new, reject_streak_new, easy_accept_streak_new,
                    step_idx + 1, save_idx, ys, ts, dts, accs, fails, codes,
                )

            loop_carry = (
                t0, flat_state0, base_dt, jnp.asarray(False), jnp.asarray(False), jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32), jnp.asarray(1.0, dtype=dtype), jnp.tile(flat_rhs(t0, flat_state0), 3),
                jnp.asarray(0.0, dtype=dtype), jnp.zeros((state_dim, state_dim), dtype=dtype), jnp.asarray(False),
                jnp.asarray(0.0, dtype=dtype), jnp.asarray(0, dtype=jnp.int32),
                real_lu0, real_piv0, complex_lu0, complex_piv0, jnp.asarray(False), jnp.asarray(0.0, dtype=dtype),
                jnp.asarray(False), jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32), jnp.asarray(1, dtype=jnp.int32),
                ys_saved, ts_saved, dts_saved, accepted_mask_saved, failed_mask_saved, fail_codes_saved,
            )
            (
                t_f, y_f_flat, dt_last, done_f, failed_f, fail_code_f, n_acc_f,
                _prev_err_f, _prev_stages_f, _prev_dt_f,
                _jacobian_f, _cache_valid_f, _cache_dt_f, _cache_age_f,
                _real_lu_f, _real_piv_f, _complex_lu_f, _complex_piv_f, _factor_cache_valid_f, _factor_cache_dt_f,
                _rejected_last_f, _reject_streak_f, _easy_accept_streak_f,
                _, _save_idx_f, ys_saved, ts_saved, dts_saved, accepted_mask_saved, failed_mask_saved, fail_codes_saved,
            ) = jax.lax.while_loop(cond_fun, body_fun, loop_carry)
            return _finalize_custom_solver_output(
                ys_saved, ts_saved, dts_saved, accepted_mask_saved, failed_mask_saved, fail_codes_saved,
                y_f_flat, t_f, done_f, failed_f, fail_code_f, n_acc_f, unpack_flat, state, species,
                temperature_active_mask=temperature_active_mask,
                fixed_temperature_profile=fixed_temperature_profile,
            )

        def cond_fun(loop_carry):
            t_value, _, _, done, failed, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, step_idx = loop_carry
            active = jnp.logical_not(jnp.logical_or(done, failed))
            return jnp.logical_and(step_idx < max_total_steps, active)

        def body_fun(loop_carry):
            (
                t_value, flat_y, dt_value, done, failed, fail_code, n_accepted, prev_error, prev_stages, prev_dt,
                jacobian_cache, cache_valid, cache_dt, cache_age,
                real_lu_cache, real_piv_cache, complex_lu_cache, complex_piv_cache, factor_cache_valid, factor_cache_dt,
                rejected_last, reject_streak, easy_accept_streak, step_idx,
            ) = loop_carry
            (
                t_new, y_new, dt_next, done_new, failed_new, fail_code_new, n_acc_new,
                prev_error_new, prev_stages_new, prev_dt_new,
                jacobian_new, cache_valid_new, cache_dt_new, cache_age_new,
                real_lu_new, real_piv_new, complex_lu_new, complex_piv_new, factor_cache_valid_new, factor_cache_dt_new,
                rejected_last_new, reject_streak_new, easy_accept_streak_new,
            ), _ = step_fn(
                (
                    t_value, flat_y, dt_value, done, failed, fail_code, n_accepted,
                    prev_error, prev_stages, prev_dt,
                    jacobian_cache, cache_valid, cache_dt, cache_age,
                    real_lu_cache, real_piv_cache, complex_lu_cache, complex_piv_cache, factor_cache_valid, factor_cache_dt,
                    rejected_last, reject_streak, easy_accept_streak,
                ),
                None,
            )
            return (
                t_new, y_new, dt_next, done_new, failed_new, fail_code_new, n_acc_new,
                prev_error_new, prev_stages_new, prev_dt_new,
                jacobian_new, cache_valid_new, cache_dt_new, cache_age_new,
                real_lu_new, real_piv_new, complex_lu_new, complex_piv_new, factor_cache_valid_new, factor_cache_dt_new,
                rejected_last_new, reject_streak_new, easy_accept_streak_new, step_idx + 1,
            )

        loop_carry = (
            t0, flat_state0, base_dt, jnp.asarray(False), jnp.asarray(False), jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32), jnp.asarray(1.0, dtype=dtype), jnp.tile(flat_rhs(t0, flat_state0), 3),
            jnp.asarray(0.0, dtype=dtype), jnp.zeros((state_dim, state_dim), dtype=dtype), jnp.asarray(False),
            jnp.asarray(0.0, dtype=dtype), jnp.asarray(0, dtype=jnp.int32),
            real_lu0, real_piv0, complex_lu0, complex_piv0, jnp.asarray(False), jnp.asarray(0.0, dtype=dtype),
            jnp.asarray(False), jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32),
        )
        (
            t_f, y_f_flat, dt_last, done_f, failed_f, fail_code_f, n_acc_f,
            _prev_err_f, _prev_stages_f, _prev_dt_f,
            _jacobian_f, _cache_valid_f, _cache_dt_f, _cache_age_f,
            _real_lu_f, _real_piv_f, _complex_lu_f, _complex_piv_f, _factor_cache_valid_f, _factor_cache_dt_f,
            _rejected_last_f, _reject_streak_f, _easy_accept_streak_f, _,
        ) = jax.lax.while_loop(cond_fun, body_fun, loop_carry)
        ys_saved_flat = jnp.expand_dims(y_f_flat, axis=0)
        ts_saved = jnp.expand_dims(t_f, axis=0)
        dts_saved = jnp.expand_dims(dt_last, axis=0)
        accepted_mask_saved = jnp.expand_dims(jnp.logical_not(failed_f), axis=0)
        failed_mask_saved = jnp.expand_dims(failed_f, axis=0)
        fail_codes_saved = jnp.expand_dims(fail_code_f, axis=0)
        return _finalize_custom_solver_output(
            ys_saved_flat, ts_saved, dts_saved, accepted_mask_saved, failed_mask_saved, fail_codes_saved,
            y_f_flat, t_f, done_f, failed_f, fail_code_f, n_acc_f, unpack_flat, state, species,
            temperature_active_mask=temperature_active_mask,
            fixed_temperature_profile=fixed_temperature_profile,
        )




def build_time_solver(solver_parameters: Any, solver_override: Any = None) -> TransportSolver:
    """Create a time solver backend from runtime parameters/config.

    `solver_override` can be either:
      - an instance with `.solve(...)` (used directly), or
      - a diffrax solver instance (wrapped in DiffraxSolver).
    """
    import diffrax

    t0 = float(solver_parameters["t0"])
    t1 = float(solver_parameters["t_final"])
    dt = float(solver_parameters["dt"])

    if solver_override is not None:
        if hasattr(solver_override, "solve"):
            return solver_override
        return DiffraxSolver(integrator=lambda: solver_override, t0=t0, t1=t1, dt=dt)

    _, backend = _select_solver_family_and_backend(solver_parameters)
    save_n = solver_parameters.get("save_n", solver_parameters.get("n_save"))
    if backend == "radau":
        return RADAUSolver(
            t0=t0,
            t1=t1,
            dt=dt,
            rtol=float(solver_parameters.get("rtol", 1.0e-6)),
            atol=float(solver_parameters.get("atol", 1.0e-8)),
            max_step=float(solver_parameters.get("max_step", max(t1 - t0, dt))),
            min_step=float(solver_parameters.get("min_step", 1.0e-14)),
            tol=float(solver_parameters.get("nonlinear_solver_tol", solver_parameters.get("tol", 1.0e-8))),
            maxiter=int(solver_parameters.get("nonlinear_solver_maxiter", solver_parameters.get("maxiter", 20))),
            linear_solver=str(solver_parameters.get("radau_linear_solver", "gmres")),
            gmres_tol=float(solver_parameters.get("radau_gmres_tol", 1.0e-8)),
            gmres_maxiter=int(solver_parameters.get("radau_gmres_maxiter", 80)),
            error_estimator=str(solver_parameters.get("radau_error_estimator", "embedded2")),
            newton_strategy=str(solver_parameters.get("radau_newton_strategy", "simplified")),
            safety_factor=float(solver_parameters.get("safety_factor", 0.9)),
            min_step_factor=float(solver_parameters.get("min_step_factor", 0.1)),
            max_step_factor=float(solver_parameters.get("max_step_factor", 5.0)),
            save_n=save_n,
        )
    if backend == "rosenbrock":
        return RosenbrockSolver(
            t0=t0,
            t1=t1,
            dt=dt,
            rtol=float(solver_parameters.get("rtol", 1.0e-6)),
            atol=float(solver_parameters.get("atol", 1.0e-8)),
            max_step=float(solver_parameters.get("max_step", max(t1 - t0, dt))),
            min_step=float(solver_parameters.get("min_step", 1.0e-14)),
            linear_solver=str(solver_parameters.get("rosenbrock_linear_solver", "gmres")),
            gmres_tol=float(solver_parameters.get("rosenbrock_gmres_tol", 1.0e-8)),
            gmres_maxiter=int(solver_parameters.get("rosenbrock_gmres_maxiter", 120)),
            safety_factor=float(solver_parameters.get("safety_factor", 0.9)),
            min_step_factor=float(solver_parameters.get("min_step_factor", 0.2)),
            max_step_factor=float(solver_parameters.get("max_step_factor", 5.0)),
            gamma=float(solver_parameters.get("rosenbrock_gamma", 1.7071067811865475)),
            save_n=save_n,
        )
    integrator_ctor = _get_diffrax_integrator(backend)
    ts_list = solver_parameters.get("ts_list")
    saveat = diffrax.SaveAt(ts=ts_list) if ts_list is not None else diffrax.SaveAt(t1=True)
    stepsize_controller = diffrax.PIDController(
        pcoeff=0.3,
        icoeff=0.4,
        rtol=float(solver_parameters.get("rtol")),
        atol=float(solver_parameters.get("atol")),
    )
    return DiffraxSolver(
        integrator=integrator_ctor,
        t0=t0,
        t1=t1,
        dt=dt,
        save_n=save_n,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=int(solver_parameters.get("max_steps", 20000)),
        throw=bool(solver_parameters.get("throw", False)),
    )




register_time_solver("diffrax_kvaerno5", lambda **kw: DiffraxSolver(_get_diffrax_integrator("diffrax_kvaerno5"), **kw))
register_time_solver("diffrax_tsit5", lambda **kw: DiffraxSolver(_get_diffrax_integrator("diffrax_tsit5"), **kw))
register_time_solver("diffrax_dopri5", lambda **kw: DiffraxSolver(_get_diffrax_integrator("diffrax_dopri5"), **kw))
register_time_solver("radau", lambda **kw: RADAUSolver(**kw))
register_time_solver("rosenbrock", lambda **kw: RosenbrockSolver(**kw))
