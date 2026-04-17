
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


TIME_SOLVER_REGISTRY: dict[str, Callable[..., "TransportSolver"]] = {}

ODE_SOLVER_BACKENDS = {
    "diffrax_kvaerno5",
    "diffrax_tsit5",
    "diffrax_dopri5",
    "radau",
    "predictor_corrector",
    "heun",
}
NONLINEAR_SOLVER_BACKENDS = {
    "newton",
    "anderson",
    "broyden",
    "jaxopt_steady_state",
}
THETA_SOLVER_BACKENDS = {
    "theta_linear",
    "theta_newton",
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
    """Pick solver family/backend while preserving legacy integrator-only configs."""
    family = str(solver_parameters.get("transport_solver_family", "auto")).strip().lower()
    backend = str(
        solver_parameters.get(
            "transport_solver_backend",
            solver_parameters.get("integrator", "diffrax_kvaerno5"),
        )
    ).strip().lower()

    if family in ("auto", ""):
        if backend in ODE_SOLVER_BACKENDS:
            family = "ode"
        elif backend in THETA_SOLVER_BACKENDS:
            family = "theta"
        elif backend in NONLINEAR_SOLVER_BACKENDS:
            family = "nonlinear"
        else:
            raise ValueError(
                f"Unknown transport solver backend '{backend}'. "
                f"Expected one of {sorted(ODE_SOLVER_BACKENDS | THETA_SOLVER_BACKENDS | NONLINEAR_SOLVER_BACKENDS)}."
            )

    if family in ("ode", "time", "time_integration"):
        return "ode", backend
    if family in ("theta", "theta_method", "torax_like"):
        return "theta", backend
    if family in ("nonlinear", "steady_state", "root"):
        return "nonlinear", backend

    raise ValueError(
        f"Unknown transport solver family '{family}'. Expected one of "
        "['auto', 'ode', 'theta', 'theta_method', 'torax_like', 'nonlinear', 'steady_state', 'root']."
    )


def _tree_add(a: Any, b: Any) -> Any:
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def _tree_sub(a: Any, b: Any) -> Any:
    return jax.tree_util.tree_map(lambda x, y: x - y, a, b)


def _tree_scale(a: Any, alpha: float | jax.Array) -> Any:
    return jax.tree_util.tree_map(lambda x: alpha * x, a)


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


def _pack_transport_state_arrays(state: Any) -> Any:
    """Convert a TransportState-like object into an array-only pytree."""
    if dataclasses.is_dataclass(state) and hasattr(state, "density") and hasattr(state, "temperature") and hasattr(state, "Er"):
        return (state.density, state.temperature, state.Er)
    return state


def _unpack_transport_state_arrays(state_like: Any, template_state: Any) -> Any:
    """Rebuild a TransportState-like object from an array-only pytree."""
    if (
        dataclasses.is_dataclass(template_state)
        and hasattr(template_state, "density")
        and hasattr(template_state, "temperature")
        and hasattr(template_state, "Er")
        and isinstance(state_like, tuple)
        and len(state_like) == 3
    ):
        density, temperature, er = state_like
        return dataclasses.replace(template_state, density=density, temperature=temperature, Er=er)
    return state_like


def _restore_state_metadata(state_like: Any, template_state: Any) -> Any:
    """Hook for restoring state metadata after solver calls."""
    return state_like


def _apply_quasi_neutrality_output(state_like: Any, species: Any, reference_state: Any) -> Any:
    """Apply quasi-neutrality to either a single state or a saved time-series of states."""
    from ._transport_equations import enforce_quasi_neutrality

    density = getattr(state_like, "density", None)
    ref_density = getattr(reference_state, "density", None)
    if density is None or ref_density is None:
        return state_like

    if density.ndim == ref_density.ndim + 1:
        out = jax.vmap(lambda s: enforce_quasi_neutrality(s, species))(state_like)
        return _restore_state_metadata(out, reference_state)
    out = enforce_quasi_neutrality(state_like, species)
    return _restore_state_metadata(out, reference_state)


def _radau_stage_matvec(
    v_flat: jax.Array,
    stage_linears: tuple[Callable[[jax.Array], jax.Array], ...],
    a_matrix: jax.Array,
    h_value: jax.Array,
    state_dim: int,
) -> jax.Array:
    """Matrix-free Jacobian-vector product for the 3-stage Radau Newton system."""
    v_stages = v_flat.reshape((3, state_dim))
    coupled = h_value * (a_matrix @ v_stages)
    out = tuple(v_stages[i] - stage_linears[i](coupled[i]) for i in range(3))
    return jnp.stack(out, axis=0).reshape((-1,))


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
        # Adapt NEOPAX vector fields to Diffrax's expected signature (t, y, args).
        import diffrax
        import equinox as eqx
        solver_state_template = _strip_state_metadata_for_solver(state)
        solver_state = _pack_transport_state_arrays(solver_state_template)

        # Diffrax expects vf(t, y, args). Capture NEOPAX runtime args in the
        # closure instead of passing non-array objects through Diffrax/Eqinox
        # loop state.
        def wrapped_vector_field(t, y, _vf_args):
            y_state = _unpack_transport_state_arrays(y, solver_state_template)
            dy_state = vector_field(t, y_state, *args)
            dy_state = _strip_state_metadata_for_solver(dy_state)
            return _pack_transport_state_arrays(dy_state)

        term = diffrax.ODETerm(wrapped_vector_field)
        solver = self.integrator()
        call_kwargs = dict(self.integrator_kwargs)
        call_kwargs.update(kwargs)
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
                y0=solver_state,
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
                y0=solver_state,
                args=None,
                **call_kwargs
            )
        # Enforce quasi-neutrality on all saved states and final state
        species = _extract_species_from_args(args)
        if species is not None:
            if hasattr(sol, 'ys'):
                ys_state = _unpack_transport_state_arrays(sol.ys, solver_state_template)
                sol = eqx.tree_at(
                    lambda s: s.ys,
                    sol,
                    _apply_quasi_neutrality_output(ys_state, species, state),
                )
        else:
            if hasattr(sol, 'ys'):
                ys_state = _unpack_transport_state_arrays(sol.ys, solver_state_template)
                sol = eqx.tree_at(
                    lambda s: s.ys,
                    sol,
                    _restore_state_metadata(ys_state, state),
                )
        return sol

@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class NewtonSolver(TransportSolver):
    tol: float = 1e-8
    maxiter: int = 50
    stop_fn: Callable | None = None

    def solve(self, state, vector_field: Callable, *args, **kwargs):
        # Steady-state: find root of vector_field(state) = 0 using JAX's lax.while_loop
        residual_fn = _as_state_residual(vector_field)
        flat_x0, unravel = jax.flatten_util.ravel_pytree(state)
        def body_fn(val):
            i, x, fx, res_norm = val
            jac = jax.jacfwd(lambda y: jax.flatten_util.ravel_pytree(residual_fn(unravel(y), *args, **kwargs))[0])(x)
            dx = jax.scipy.linalg.solve(jac, -fx)
            x_new = x + dx
            fx_new = jax.flatten_util.ravel_pytree(residual_fn(unravel(x_new), *args, **kwargs))[0]
            res_norm_new = jnp.linalg.norm(fx_new)
            return (i+1, x_new, fx_new, res_norm_new)

        def cond_fn(val):
            i, x, fx, res_norm = val
            if self.stop_fn is not None:
                # stop_fn should return True to continue, False to stop
                return self.stop_fn(i, x, fx, res_norm)
            return jnp.logical_and(res_norm > self.tol, i < self.maxiter)

        fx0 = jax.flatten_util.ravel_pytree(residual_fn(state, *args, **kwargs))[0]
        res_norm0 = jnp.linalg.norm(fx0)
        init_val = (0, flat_x0, fx0, res_norm0)
        i_final, x_final, fx_final, res_norm_final = jax.lax.while_loop(cond_fn, body_fn, init_val)
        # Optionally, return a result object for consistency
        result = {
            'solution': unravel(x_final),
            'iterations': i_final,
            'residual_norm': res_norm_final,
            'converged': res_norm_final <= self.tol
        }
        return result


# Jaxopt-based steady-state solver (minimize norm of residual)
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class JaxoptSteadyStateSolver(TransportSolver):
    tol: float = 1e-8
    maxiter: int = 100
    optimizer: Any = None

    def solve(self, state, vector_field: Callable, *args, **kwargs):
        # Minimize the L2 norm of the residual: ||vector_field(state)||^2
        residual_fn = _as_state_residual(vector_field)
        optimizer_cls = self.optimizer
        if optimizer_cls is None:
            import jaxopt

            optimizer_cls = jaxopt.ProjectedGradient
        def loss_fn(x):
            res = residual_fn(x, *args, **kwargs)
            flat, _ = jax.flatten_util.ravel_pytree(res)
            return jnp.sum(flat ** 2)
        opt = optimizer_cls(fun=loss_fn, maxiter=self.maxiter, tol=self.tol)
        flat_state, unravel = jax.flatten_util.ravel_pytree(state)
        sol = opt.run(flat_state)
        final_state = unravel(sol.params)
        final_res = residual_fn(final_state, *args, **kwargs)
        flat_res, _ = jax.flatten_util.ravel_pytree(final_res)
        result = {
            'solution': final_state,
            'iterations': sol.state.iter_num if hasattr(sol.state, 'iter_num') else self.maxiter,
            'residual_norm': jnp.linalg.norm(flat_res),
            'converged': jnp.linalg.norm(flat_res) <= self.tol
        }
        return result
    


# Anderson and Broyden solvers (JAX, torax-style)
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class AndersonSolver(TransportSolver):
    m: int = 5  # history size
    tol: float = 1e-8
    maxiter: int = 100
    stop_fn: Callable | None = None

    def solve(self, state, vector_field: Callable, *args, **kwargs):
        # Flatten state for easier history management
        residual_fn = _as_state_residual(vector_field)
        flat_x0, unravel = jax.flatten_util.ravel_pytree(state)
        dim = flat_x0.shape[0]
        m = self.m
        def body_fn(val):
            i, x, fx, X_hist, F_hist, res_norm = val
            X_hist = jnp.roll(X_hist, -1, axis=0).at[-1].set(x)
            F_hist = jnp.roll(F_hist, -1, axis=0).at[-1].set(fx)
            dX = X_hist - X_hist[0]
            dF = F_hist - F_hist[0]
            G = dF[1:] - dF[:-1]
            G = G.reshape((m-1, dim))
            rhs = -dF[-1]
            # Use least squares for mixing coefficients
            gamma = jnp.linalg.lstsq(G.T, rhs, rcond=None)[0] if m > 1 else jnp.zeros(m-1)
            x_new = x + fx + dX[1:].T @ gamma
            fx_new = jax.flatten_util.ravel_pytree(residual_fn(unravel(x_new), *args, **kwargs))[0]
            res_norm_new = jnp.linalg.norm(fx_new)
            return (i+1, x_new, fx_new, X_hist, F_hist, res_norm_new)

        def cond_fn(val):
            i, x, fx, X_hist, F_hist, res_norm = val
            if self.stop_fn is not None:
                return self.stop_fn(i, x, fx, res_norm)
            return jnp.logical_and(res_norm > self.tol, i < self.maxiter)

        X_hist = jnp.tile(flat_x0, (self.m, 1))
        fx0 = jax.flatten_util.ravel_pytree(residual_fn(state, *args, **kwargs))[0]
        F_hist = jnp.tile(fx0, (self.m, 1))
        res_norm0 = jnp.linalg.norm(fx0)
        init_val = (0, flat_x0, fx0, X_hist, F_hist, res_norm0)
        i_final, x_final, fx_final, _, _, res_norm_final = jax.lax.while_loop(cond_fn, body_fn, init_val)
        result = {
            'solution': unravel(x_final),
            'iterations': i_final,
            'residual_norm': res_norm_final,
            'converged': res_norm_final <= self.tol
        }
        return result


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class BroydenSolver(TransportSolver):
    tol: float = 1e-8
    maxiter: int = 100
    stop_fn: Callable | None = None

    def solve(self, state, vector_field: Callable, *args, **kwargs):
        residual_fn = _as_state_residual(vector_field)
        flat_x0, unravel = jax.flatten_util.ravel_pytree(state)
        fx0 = jax.flatten_util.ravel_pytree(residual_fn(state, *args, **kwargs))[0]
        n = flat_x0.shape[0]
        B = jnp.eye(n)
        def body_fn(val):
            i, x, fx, B, res_norm = val
            dx = -jax.scipy.linalg.solve(B, fx)
            x_new = x + dx
            fx_new = jax.flatten_util.ravel_pytree(residual_fn(unravel(x_new), *args, **kwargs))[0]
            y = fx_new - fx
            dx_col = dx[:, None]
            dy_col = (y - B @ dx)[:, None]
            B_new = B + dy_col @ dx_col.T / (dx_col.T @ dx_col + 1e-12)
            res_norm_new = jnp.linalg.norm(fx_new)
            return (i+1, x_new, fx_new, B_new, res_norm_new)

        def cond_fn(val):
            i, x, fx, B, res_norm = val
            if self.stop_fn is not None:
                return self.stop_fn(i, x, fx, res_norm)
            return jnp.logical_and(res_norm > self.tol, i < self.maxiter)

        init_val = (0, flat_x0, fx0, B, jnp.linalg.norm(fx0))
        i_final, x_final, fx_final, _, res_norm_final = jax.lax.while_loop(cond_fn, body_fn, init_val)
        result = {
            'solution': unravel(x_final),
            'iterations': i_final,
            'residual_norm': res_norm_final,
            'converged': res_norm_final <= self.tol
        }
        return result


# Predictor-Corrector (Heun's method) integrator, JAX-native
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class PredictorCorrectorSolver(TransportSolver):
    t0: float
    t1: float
    dt: float
    n_steps: int = 0  # computed at construction; 0 means use dt directly
    tol: float = 1e-8
    maxiter: int = 100

    def __init__(self, t0: float, t1: float, dt: float, tol: float = 1e-8, maxiter: int = 100, save_n=None):
        n_steps = int(jnp.ceil((float(t1) - float(t0)) / float(dt)))
        object.__setattr__(self, 't0', t0)
        object.__setattr__(self, 't1', t1)
        object.__setattr__(self, 'dt', dt)
        object.__setattr__(self, 'n_steps', n_steps)
        object.__setattr__(self, 'tol', tol)
        object.__setattr__(self, 'maxiter', maxiter)
        object.__setattr__(self, 'save_n', save_n)

    def solve(self, state, vector_field: Callable, *args, **kwargs):
        # Implements Heun's method (predictor-corrector, explicit trapezoidal)
        from ._transport_equations import enforce_quasi_neutrality
        n_steps = self.n_steps
        dt = (self.t1 - self.t0) / n_steps
        species = _extract_species_from_args(args)
        def step_fn(val, _):
            t, y = val
            f0 = vector_field(t, y, *args, **kwargs)
            y_pred = jax.tree_util.tree_map(lambda yi, fi: yi + dt * fi, y, f0)
            f1 = vector_field(t + dt, y_pred, *args, **kwargs)
            y_next = jax.tree_util.tree_map(lambda yi, fi0, fi1: yi + 0.5 * dt * (fi0 + fi1), y, f0, f1)
            return (t + dt, y_next), y_next
        (tf, yf), ys = jax.lax.scan(step_fn, (self.t0, state), None, length=n_steps)
        save_n = getattr(self, 'save_n', None)
        if save_n is not None and save_n > 1:
            if species is not None:
                ys = jax.vmap(lambda s: enforce_quasi_neutrality(s, species))(ys)
                yf = enforce_quasi_neutrality(yf, species)
            idxs = jnp.linspace(0, n_steps-1, save_n).round().astype(int)
            ys_saved = ys[idxs]
        else:
            if species is not None:
                yf = enforce_quasi_neutrality(yf, species)
            ys_saved = _save_state_series(yf)
        result = {
            'ys': ys_saved,
            't0': self.t0,
            't1': self.t1,
            'dt': dt,
            'n_steps': n_steps,
            'final_state': yf
        }
        return result


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class ThetaLinearSolver(TransportSolver):
    """Theta-method with predictor-corrector fixed-point iterations per time step."""

    t0: float
    t1: float
    dt: float
    theta_implicit: float = 1.0
    use_predictor_corrector: bool = True
    n_corrector_steps: int = 1
    n_steps: int = 0

    def __init__(
        self,
        t0: float,
        t1: float,
        dt: float,
        theta_implicit: float = 1.0,
        use_predictor_corrector: bool = True,
        n_corrector_steps: int = 1,
        save_n=None,
    ):
        n_steps = max(1, int(jnp.ceil((float(t1) - float(t0)) / float(dt))))
        object.__setattr__(self, "t0", t0)
        object.__setattr__(self, "t1", t1)
        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "theta_implicit", theta_implicit)
        object.__setattr__(self, "use_predictor_corrector", use_predictor_corrector)
        object.__setattr__(self, "n_corrector_steps", int(max(0, n_corrector_steps)))
        object.__setattr__(self, "n_steps", n_steps)
        object.__setattr__(self, "save_n", save_n)

    def solve(self, state, vector_field: Callable, *args, **kwargs):
        n_steps = self.n_steps
        dt = (self.t1 - self.t0) / n_steps
        theta = self.theta_implicit

        from ._transport_equations import enforce_quasi_neutrality
        species = _extract_species_from_args(args)
        def one_step(carry, _):
            t, y = carry
            f_old = vector_field(t, y, *args, **kwargs)
            y_guess = _tree_add(y, _tree_scale(f_old, dt))

            def fixed_point_map(y_new):
                f_new = vector_field(t + dt, y_new, *args, **kwargs)
                rhs = _tree_add(
                    _tree_scale(f_old, 1.0 - theta),
                    _tree_scale(f_new, theta),
                )
                return _tree_add(y, _tree_scale(rhs, dt))

            if self.use_predictor_corrector and self.n_corrector_steps > 0:
                def body(_, y_iter):
                    return fixed_point_map(y_iter)

                y_next = jax.lax.fori_loop(0, self.n_corrector_steps, body, y_guess)
            else:
                y_next = fixed_point_map(y_guess)

            return (t + dt, y_next), y_next

        save_n = getattr(self, "save_n", None)
        (tf, yf), ys = jax.lax.scan(one_step, (self.t0, state), None, length=n_steps)
        if save_n is not None and save_n > 1:
            if species is not None:
                ys = jax.vmap(lambda s: enforce_quasi_neutrality(s, species))(ys)
                yf = enforce_quasi_neutrality(yf, species)
            idxs = jnp.linspace(0, n_steps - 1, save_n).round().astype(int)
            ys_saved = ys[idxs]
        else:
            if species is not None:
                yf = enforce_quasi_neutrality(yf, species)
            ys_saved = _save_state_series(yf)
        return {
            "ys": ys_saved,
            "t0": self.t0,
            "t1": self.t1,
            "dt": dt,
            "n_steps": n_steps,
            "final_state": yf,
            "time": tf,
        }


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class ThetaNewtonSolver(TransportSolver):
    """Theta-method with Newton solve of the implicit per-step residual."""

    t0: float
    t1: float
    dt: float
    theta_implicit: float = 1.0
    tol: float = 1e-8
    maxiter: int = 50
    ptc_enabled: bool = True
    ptc_dt_min_factor: float = 1.0e-4
    ptc_dt_max_factor: float = 1.0e3
    ptc_growth: float = 1.5
    ptc_shrink: float = 0.5
    line_search_enabled: bool = True
    line_search_contraction: float = 0.5
    line_search_min_alpha: float = 1.0e-4
    line_search_c: float = 1.0e-4
    max_step_retries: int = 8
    linear_solver: str = "direct"
    gmres_tol: float = 1.0e-8
    gmres_maxiter: int = 200
    trust_region_enabled: bool = False
    trust_radius: float = 1.0
    homotopy_steps: int = 1
    differentiable_mode: bool = False
    n_steps: int = 0

    def __init__(
        self,
        t0: float,
        t1: float,
        dt: float,
        theta_implicit: float = 1.0,
        tol: float = 1e-8,
        maxiter: int = 50,
        ptc_enabled: bool = True,
        ptc_dt_min_factor: float = 1.0e-4,
        ptc_dt_max_factor: float = 1.0e3,
        ptc_growth: float = 1.5,
        ptc_shrink: float = 0.5,
        line_search_enabled: bool = True,
        line_search_contraction: float = 0.5,
        line_search_min_alpha: float = 1.0e-4,
        line_search_c: float = 1.0e-4,
        max_step_retries: int = 8,
        linear_solver: str = "direct",
        gmres_tol: float = 1.0e-8,
        gmres_maxiter: int = 200,
        trust_region_enabled: bool = False,
        trust_radius: float = 1.0,
        homotopy_steps: int = 1,
        differentiable_mode: bool = False,
        save_n=None,
    ):
        n_steps = max(1, int(jnp.ceil((float(t1) - float(t0)) / float(dt))))
        object.__setattr__(self, "t0", t0)
        object.__setattr__(self, "t1", t1)
        object.__setattr__(self, "dt", dt)
        object.__setattr__(self, "theta_implicit", theta_implicit)
        object.__setattr__(self, "tol", tol)
        object.__setattr__(self, "maxiter", maxiter)
        object.__setattr__(self, "ptc_enabled", bool(ptc_enabled))
        object.__setattr__(self, "ptc_dt_min_factor", float(ptc_dt_min_factor))
        object.__setattr__(self, "ptc_dt_max_factor", float(ptc_dt_max_factor))
        object.__setattr__(self, "ptc_growth", float(ptc_growth))
        object.__setattr__(self, "ptc_shrink", float(ptc_shrink))
        object.__setattr__(self, "line_search_enabled", bool(line_search_enabled))
        object.__setattr__(self, "line_search_contraction", float(line_search_contraction))
        object.__setattr__(self, "line_search_min_alpha", float(line_search_min_alpha))
        object.__setattr__(self, "line_search_c", float(line_search_c))
        object.__setattr__(self, "max_step_retries", int(max(0, max_step_retries)))
        object.__setattr__(self, "linear_solver", str(linear_solver).strip().lower())
        object.__setattr__(self, "gmres_tol", float(gmres_tol))
        object.__setattr__(self, "gmres_maxiter", int(max(1, gmres_maxiter)))
        object.__setattr__(self, "trust_region_enabled", bool(trust_region_enabled))
        object.__setattr__(self, "trust_radius", float(trust_radius))
        object.__setattr__(self, "homotopy_steps", int(max(1, homotopy_steps)))
        object.__setattr__(self, "differentiable_mode", bool(differentiable_mode))
        object.__setattr__(self, "n_steps", n_steps)
        object.__setattr__(self, "save_n", save_n)

    def solve(self, state, vector_field: Callable, *args, **kwargs):
        ap_preconditioner = kwargs.pop("ap_preconditioner", None)

        def _sanitize_precond(diag, flat_ref):
            if diag is None:
                return jnp.zeros_like(flat_ref)
            arr = jnp.asarray(diag, dtype=flat_ref.dtype).reshape(-1)
            n = flat_ref.shape[0]
            m = arr.shape[0]
            arr = jax.lax.cond(
                m >= n,
                lambda _: arr[:n],
                lambda _: jnp.pad(arr, (0, n - m), mode="constant"),
                operand=None,
            )
            return jnp.maximum(arr, 0.0)

        n_steps = self.n_steps
        base_dt = jnp.asarray((self.t1 - self.t0) / n_steps)
        theta = jnp.asarray(self.theta_implicit)
        tol = jnp.asarray(self.tol)
        t_final = jnp.asarray(self.t1)
        dt_min = jnp.asarray(max(float((self.t1 - self.t0) / n_steps) * max(self.ptc_dt_min_factor, 1.0e-12), 1.0e-14))
        dt_max = jnp.asarray(max(float((self.t1 - self.t0) / n_steps) * max(self.ptc_dt_max_factor, 1.0), float(dt_min)))
        max_total_steps = int(max(1, n_steps * max(1, self.max_step_retries + 1)))

        y0 = state
        flat0, unravel = jax.flatten_util.ravel_pytree(y0)
        del flat0

        if self.differentiable_mode:
            # Autodiff-friendly path: fixed step count and fixed Newton/homotopy
            # iterations, no accept/reject branching.
            dt = base_dt

            def one_step(carry, _):
                t, y = carry
                f_old = vector_field(t, y, *args, **kwargs)
                y_guess = _tree_add(y, _tree_scale(f_old, dt))
                flat_guess, _ = jax.flatten_util.ravel_pytree(y_guess)

                def homotopy_body(i, flat_in):
                    lam = jnp.asarray((i + 1) / self.homotopy_steps)

                    def residual_flat(flat_ynew):
                        y_new = unravel(flat_ynew)
                        f_new = vector_field(t + dt, y_new, *args, **kwargs)
                        f_h = _tree_add(_tree_scale(f_old, 1.0 - lam), _tree_scale(f_new, lam))
                        rhs = _tree_add(_tree_scale(f_old, 1.0 - theta), _tree_scale(f_h, theta))
                        implicit_update = _tree_add(y, _tree_scale(rhs, dt))
                        r_tree = _tree_sub(y_new, implicit_update)
                        r_flat, _ = jax.flatten_util.ravel_pytree(r_tree)
                        return r_flat

                    def newton_body(_, flat_cur):
                        r = residual_flat(flat_cur)
                        ap_diag = jnp.zeros_like(flat_cur)
                        if ap_preconditioner is not None:
                            ap_diag = _sanitize_precond(
                                ap_preconditioner(t + dt, unravel(flat_cur), *args),
                                flat_cur,
                            )

                        if self.linear_solver == "gmres":
                            _, residual_lin = jax.linearize(residual_flat, flat_cur)

                            def matvec(v):
                                return residual_lin(v) + ap_diag * v

                            delta, _ = jax.scipy.sparse.linalg.gmres(
                                matvec,
                                -r,
                                tol=self.gmres_tol,
                                atol=self.gmres_tol,
                                maxiter=self.gmres_maxiter,
                            )
                        else:
                            jac = jax.jacfwd(residual_flat)(flat_cur)
                            jac = jac + jnp.diag(ap_diag)
                            delta = jax.scipy.linalg.solve(jac, -r)

                        finite = jnp.all(jnp.isfinite(delta))
                        delta = jnp.where(finite, delta, jnp.zeros_like(delta))

                        if self.trust_region_enabled:
                            dnorm = jnp.linalg.norm(delta)
                            # Smooth trust-region damping to avoid hard clipping kinks.
                            scale = self.trust_radius / jnp.sqrt(self.trust_radius * self.trust_radius + dnorm * dnorm + 1.0e-30)
                            delta = delta * scale

                        if self.line_search_enabled:
                            rnorm = jnp.linalg.norm(r)
                            trial = residual_flat(flat_cur + delta)
                            trial_norm = jnp.linalg.norm(trial)
                            # Smooth acceptance proxy in (0,1], favoring descent steps.
                            ratio = (rnorm - trial_norm) / (rnorm + 1.0e-30)
                            alpha_soft = jax.nn.sigmoid(10.0 * ratio)
                            alpha = self.line_search_min_alpha + (1.0 - self.line_search_min_alpha) * alpha_soft
                        else:
                            alpha = jnp.asarray(1.0)

                        return flat_cur + alpha * delta

                    return jax.lax.fori_loop(0, self.maxiter, newton_body, flat_in)

                flat_sol = jax.lax.fori_loop(0, self.homotopy_steps, homotopy_body, flat_guess)
                y_next = unravel(flat_sol)
                t_next = t + dt
                out = (y_next, t_next, dt, jnp.asarray(True), jnp.asarray(False), jnp.asarray(0))
                return (t_next, y_next), out

            save_n = getattr(self, 'save_n', None)
            if save_n is not None and save_n > 1:
                (t_f, y_f), outs = jax.lax.scan(
                    one_step,
                    (jnp.asarray(self.t0), y0),
                    None,
                    length=n_steps,
                )
                ys_all, ts_all, dts_all, accepted_mask, failed_mask, fail_codes = outs
                idxs = jnp.linspace(0, n_steps-1, save_n).round().astype(int)
                ys_saved = ys_all[idxs]
                ts_saved = ts_all[idxs]
                dts_saved = dts_all[idxs]
                accepted_mask_saved = accepted_mask[idxs]
                failed_mask_saved = failed_mask[idxs]
                fail_codes_saved = fail_codes[idxs]
            else:
                def body(i, carry):
                    t, y = carry
                    (t_next, y_next), _ = one_step((t, y), None)
                    return t_next, y_next

                t_f, y_f = jax.lax.fori_loop(0, n_steps, body, (jnp.asarray(self.t0), y0))
                ys_saved = _save_state_series(y_f)
                ts_saved = _save_scalar_series(t_f)
                dts_saved = _save_scalar_series(dt)
                accepted_mask_saved = _save_scalar_series(True)
                failed_mask_saved = _save_scalar_series(False)
                fail_codes_saved = _save_scalar_series(0)
            return {
                "ys": ys_saved,
                "t0": self.t0,
                "t1": self.t1,
                "dt": dt,
                "dts": dts_saved,
                "accepted_mask": accepted_mask_saved,
                "failed_mask": failed_mask_saved,
                "fail_codes": fail_codes_saved,
                "n_steps": jnp.asarray(n_steps),
                "done": jnp.asarray(True),
                "failed": jnp.asarray(False),
                "fail_code": jnp.asarray(0),
                "final_state": y_f,
                "time": ts_saved,
                "final_time": t_f,
            }

        def _line_search(flat_y, r, rnorm, delta, residual_flat):
            alpha0 = jnp.asarray(1.0)
            init = (alpha0, jnp.asarray(False), flat_y, r, rnorm)

            def cond_fn(ls):
                alpha, accepted, *_ = ls
                return jnp.logical_and(jnp.logical_not(accepted), alpha >= self.line_search_min_alpha)

            def body_fn(ls):
                alpha, _, flat_best, r_best, rnorm_best = ls
                trial_flat = flat_y + alpha * delta
                trial_r = residual_flat(trial_flat)
                trial_norm = jnp.linalg.norm(trial_r)
                sufficient = trial_norm <= (1.0 - self.line_search_c * alpha) * rnorm

                def _accept(_):
                    return alpha, jnp.asarray(True), trial_flat, trial_r, trial_norm

                def _reject(_):
                    return alpha * self.line_search_contraction, jnp.asarray(False), flat_best, r_best, rnorm_best

                return jax.lax.cond(sufficient, _accept, _reject, operand=None)

            return jax.lax.while_loop(cond_fn, body_fn, init)

        def _newton_stage(flat_start, y_prev, f_old, t_prev, dt_trial, lam):
            def residual_flat(flat_ynew):
                y_new = unravel(flat_ynew)
                f_new = vector_field(t_prev + dt_trial, y_new, *args, **kwargs)
                f_h = _tree_add(_tree_scale(f_old, 1.0 - lam), _tree_scale(f_new, lam))
                rhs = _tree_add(_tree_scale(f_old, 1.0 - theta), _tree_scale(f_h, theta))
                implicit_update = _tree_add(y_prev, _tree_scale(rhs, dt_trial))
                r_tree = _tree_sub(y_new, implicit_update)
                r_flat, _ = jax.flatten_util.ravel_pytree(r_tree)
                return r_flat

            r0 = residual_flat(flat_start)
            init = (
                jnp.asarray(0),
                flat_start,
                r0,
                jnp.linalg.norm(r0),
                jnp.asarray(False),
                jnp.asarray(False),
            )

            def cond_fn(st):
                it, _, _, rnorm, line_fail, linear_fail = st
                not_done = jnp.logical_and(rnorm > tol, it < self.maxiter)
                ok = jnp.logical_not(jnp.logical_or(line_fail, linear_fail))
                return jnp.logical_and(not_done, ok)

            def body_fn(st):
                it, flat_y, r, rnorm, line_fail, linear_fail = st
                ap_diag = jnp.zeros_like(flat_y)
                if ap_preconditioner is not None:
                    ap_diag = _sanitize_precond(
                        ap_preconditioner(t_prev + dt_trial, unravel(flat_y), *args),
                        flat_y,
                    )

                if self.linear_solver == "gmres":
                    _, residual_lin = jax.linearize(residual_flat, flat_y)

                    def matvec(v):
                        return residual_lin(v) + ap_diag * v

                    delta, _ = jax.scipy.sparse.linalg.gmres(
                        matvec,
                        -r,
                        tol=self.gmres_tol,
                        atol=self.gmres_tol,
                        maxiter=self.gmres_maxiter,
                    )
                else:
                    jac = jax.jacfwd(residual_flat)(flat_y)
                    jac = jac + jnp.diag(ap_diag)
                    delta = jax.scipy.linalg.solve(jac, -r)

                linear_ok = jnp.all(jnp.isfinite(delta))

                def _clip(d):
                    dnorm = jnp.linalg.norm(d)
                    scale = jnp.where(dnorm > self.trust_radius, self.trust_radius / (dnorm + 1.0e-30), 1.0)
                    return d * scale

                delta = _clip(delta) if self.trust_region_enabled else delta

                def _after_linear_ok(_):
                    if self.line_search_enabled:
                        alpha, accepted_ls, flat_next, r_next, rnorm_next = _line_search(flat_y, r, rnorm, delta, residual_flat)
                        del alpha
                        line_fail_next = jnp.logical_not(accepted_ls)
                        return it + 1, flat_next, r_next, rnorm_next, line_fail_next, jnp.asarray(False)

                    flat_next = flat_y + delta
                    r_next = residual_flat(flat_next)
                    rnorm_next = jnp.linalg.norm(r_next)
                    return it + 1, flat_next, r_next, rnorm_next, jnp.asarray(False), jnp.asarray(False)

                def _after_linear_fail(_):
                    return it + 1, flat_y, r, rnorm, jnp.asarray(False), jnp.asarray(True)

                return jax.lax.cond(linear_ok, _after_linear_ok, _after_linear_fail, operand=None)

            it_f, flat_f, r_f, rnorm_f, line_fail_f, linear_fail_f = jax.lax.while_loop(cond_fn, body_fn, init)
            converged = jnp.logical_and(rnorm_f <= tol, jnp.logical_not(jnp.logical_or(line_fail_f, linear_fail_f)))
            return flat_f, converged, it_f, line_fail_f, linear_fail_f

        def _solve_step(carry):
            t, y, dt_current, done, failed, fail_code, n_acc = carry
            f_old = vector_field(t, y, *args, **kwargs)
            y_guess = _tree_add(y, _tree_scale(f_old, jnp.minimum(dt_current, t_final - t)))
            flat_guess, _ = jax.flatten_util.ravel_pytree(y_guess)
            dt_trial0 = jnp.minimum(dt_current, t_final - t)

            init_retry = (
                jnp.asarray(0),            # retry count
                dt_trial0,                 # dt trial
                jnp.asarray(False),        # accepted
                flat_guess,                # flat solution
                jnp.asarray(0),            # newton iterations total
                jnp.asarray(False),        # line-search failed
                jnp.asarray(False),        # linear solve failed
                dt_current,                # next dt suggestion
                jnp.asarray(-1),           # failed stage
            )

            def retry_cond(rs):
                retry_i, dt_trial, accepted, *_ = rs
                return jnp.logical_and(jnp.logical_not(accepted), jnp.logical_and(retry_i <= self.max_step_retries, dt_trial > dt_min * (1.0 + 1.0e-12)))

            def retry_body(rs):
                retry_i, dt_trial, _, flat_seed, _, _, _, _, _ = rs
                stage_init = (
                    flat_seed,
                    jnp.asarray(True),
                    jnp.asarray(0),
                    jnp.asarray(False),
                    jnp.asarray(False),
                    jnp.asarray(-1),
                )

                def stage_body(i, stg):
                    flat_y, ok, nit_total, line_fail, linear_fail, failed_stage = stg

                    def _do_stage(_):
                        lam = jnp.asarray((i + 1) / self.homotopy_steps)
                        flat_n, conv, nit, lfail, xfail = _newton_stage(flat_y, y, f_old, t, dt_trial, lam)
                        ok_n = jnp.logical_and(ok, conv)
                        failed_stage_n = jnp.where(jnp.logical_and(ok, jnp.logical_not(conv)), i, failed_stage)
                        return (
                            flat_n,
                            ok_n,
                            nit_total + nit,
                            jnp.logical_or(line_fail, lfail),
                            jnp.logical_or(linear_fail, xfail),
                            failed_stage_n,
                        )

                    def _skip_stage(_):
                        return stg

                    return jax.lax.cond(ok, _do_stage, _skip_stage, operand=None)

                flat_sol, stage_ok, nit_total, line_fail, linear_fail, failed_stage = jax.lax.fori_loop(
                    0,
                    self.homotopy_steps,
                    stage_body,
                    stage_init,
                )

                accepted = stage_ok

                quick_thr = max(2, self.maxiter // 6)
                slow_thr = max(6, (2 * self.maxiter) // 3)
                dt_next = jnp.where(
                    self.ptc_enabled,
                    jnp.where(
                        nit_total <= quick_thr,
                        jnp.minimum(dt_trial * self.ptc_growth, dt_max),
                        jnp.where(nit_total >= slow_thr, jnp.maximum(dt_trial * self.ptc_shrink, dt_min), dt_trial),
                    ),
                    dt_trial,
                )

                dt_retry = jnp.where(self.ptc_enabled, jnp.maximum(dt_trial * self.ptc_shrink, dt_min), dt_trial)
                dt_out = jnp.where(accepted, dt_next, dt_retry)

                return (
                    retry_i + 1,
                    dt_retry,
                    accepted,
                    flat_sol,
                    nit_total,
                    line_fail,
                    linear_fail,
                    dt_out,
                    failed_stage,
                )

            retry_final = jax.lax.while_loop(retry_cond, retry_body, init_retry)
            _, _, accepted, flat_sol, _, line_fail, linear_fail, dt_next, _ = retry_final

            def _accept(_):
                y_new = unravel(flat_sol)
                dt_used = jnp.minimum(dt_current, t_final - t)
                t_new = t + dt_used
                done_new = t_new >= (t_final - 1.0e-15)
                return (t_new, y_new, dt_next, done_new, jnp.asarray(False), fail_code, n_acc + 1), (y_new, t_new, dt_used, jnp.asarray(True), jnp.asarray(False), fail_code)

            def _reject(_):
                code = jnp.where(linear_fail, 1, jnp.where(line_fail, 2, 3))
                return (t, y, jnp.maximum(dt_current * self.ptc_shrink, dt_min), done, jnp.asarray(True), code, n_acc), (y, t, jnp.asarray(0.0), jnp.asarray(False), jnp.asarray(True), code)

            return jax.lax.cond(accepted, _accept, _reject, operand=None)

        def step_fn(carry, _):
            t, y, dt_current, done, failed, fail_code, n_acc = carry

            def _skip(_):
                return carry, (y, t, jnp.asarray(0.0), jnp.asarray(False), failed, fail_code)

            def _run(_):
                return _solve_step(carry)

            return jax.lax.cond(jnp.logical_or(done, failed), _skip, _run, operand=None)

        init_carry = (
            jnp.asarray(self.t0),
            y0,
            jnp.asarray(base_dt),
            jnp.asarray(False),
            jnp.asarray(False),
            jnp.asarray(0),
            jnp.asarray(0),
        )

        save_n = getattr(self, 'save_n', None)
        if save_n is not None and save_n > 1:
            # Precompute save times
            save_times = jnp.linspace(self.t0, self.t1, save_n)
            y0, t0, dt0, acc0, fail0, code0 = step_fn(init_carry, None)[1]
            ys_saved = jnp.zeros((save_n,) + y0.shape, dtype=y0.dtype)
            ts_saved = jnp.zeros((save_n,), dtype=t0.dtype)
            dts_saved = jnp.zeros((save_n,), dtype=dt0.dtype)
            accepted_mask_saved = jnp.zeros((save_n,), dtype=acc0.dtype)
            failed_mask_saved = jnp.zeros((save_n,), dtype=fail0.dtype)
            fail_codes_saved = jnp.zeros((save_n,), dtype=code0.dtype)
            def cond_fun(loop_carry):
                t, y, dt_current, done, failed, fail_code, n_acc, save_idx, ys, ts, dts, accs, fails, codes = loop_carry
                return jnp.logical_and(save_idx < save_n, jnp.logical_not(done))
            def body_fun(loop_carry):
                t, y, dt_current, done, failed, fail_code, n_acc, save_idx, ys, ts, dts, accs, fails, codes = loop_carry
                # Step
                (t_new, y_new, dt_next, done_new, failed_new, fail_code_new, n_acc_new), (y_out, t_out, dt_out, acc_out, fail_out, code_out) = step_fn((t, y, dt_current, done, failed, fail_code, n_acc), None)
                # Save if crossed next save time
                save_cond = jnp.logical_and(t_new >= save_times[save_idx], save_idx < save_n)
                ys = ys.at[save_idx].set(jax.lax.select(save_cond, y_new, ys[save_idx]))
                ts = ts.at[save_idx].set(jax.lax.select(save_cond, t_new, ts[save_idx]))
                dts = dts.at[save_idx].set(jax.lax.select(save_cond, dt_next, dts[save_idx]))
                accs = accs.at[save_idx].set(jax.lax.select(save_cond, acc_out, accs[save_idx]))
                fails = fails.at[save_idx].set(jax.lax.select(save_cond, fail_out, fails[save_idx]))
                codes = codes.at[save_idx].set(jax.lax.select(save_cond, code_out, codes[save_idx]))
                save_idx_new = save_idx + save_cond.astype(int)
                return (t_new, y_new, dt_next, done_new, failed_new, fail_code_new, n_acc_new, save_idx_new, ys, ts, dts, accs, fails, codes)
            # Initial carry
            loop_carry = (self.t0, y0, base_dt, False, False, 0, 0, 0, ys_saved, ts_saved, dts_saved, accepted_mask_saved, failed_mask_saved, fail_codes_saved)
            loop_carry = jax.lax.while_loop(cond_fun, body_fun, loop_carry)
            t_f, y_f, _, done_f, failed_f, fail_code_f, n_acc_f, _, ys_saved, ts_saved, dts_saved, accepted_mask_saved, failed_mask_saved, fail_codes_saved = loop_carry
        else:
            def cond_fun(loop_carry):
                t, y, dt_current, done, failed, fail_code, n_acc, step_idx = loop_carry
                active = jnp.logical_not(jnp.logical_or(done, failed))
                return jnp.logical_and(step_idx < max_total_steps, active)

            def body_fun(loop_carry):
                t, y, dt_current, done, failed, fail_code, n_acc, step_idx = loop_carry
                (t_new, y_new, dt_next, done_new, failed_new, fail_code_new, n_acc_new), _ = step_fn(
                    (t, y, dt_current, done, failed, fail_code, n_acc),
                    None,
                )
                return (t_new, y_new, dt_next, done_new, failed_new, fail_code_new, n_acc_new, step_idx + 1)

            loop_carry = (
                jnp.asarray(self.t0),
                y0,
                jnp.asarray(base_dt),
                jnp.asarray(False),
                jnp.asarray(False),
                jnp.asarray(0),
                jnp.asarray(0),
                jnp.asarray(0),
            )
            t_f, y_f, dt_last, done_f, failed_f, fail_code_f, n_acc_f, _ = jax.lax.while_loop(cond_fun, body_fun, loop_carry)
            ys_saved = _save_state_series(y_f)
            ts_saved = _save_scalar_series(t_f)
            dts_saved = _save_scalar_series(dt_last)
            accepted_mask_saved = _save_scalar_series(jnp.logical_not(failed_f))
            failed_mask_saved = _save_scalar_series(failed_f)
            fail_codes_saved = _save_scalar_series(fail_code_f)
        return {
            "ys": ys_saved,
            "t0": self.t0,
            "t1": self.t1,
            "dt": base_dt,
            "dts": dts_saved,
            "accepted_mask": accepted_mask_saved,
            "failed_mask": failed_mask_saved,
            "fail_codes": fail_codes_saved,
            "n_steps": n_acc_f,
            "done": done_f,
            "failed": failed_f,
            "fail_code": fail_code_f,
            "final_state": y_f,
            "time": ts_saved,
            "final_time": t_f,
        }

# Usage example (not executed here):
# solver = DiffraxSolver(diffrax.Tsit5, t0=0, t1=1, dt=1e-3)
# result = solver.solve(state, vector_field)


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

    family, backend = _select_solver_family_and_backend(solver_parameters)
    theta_implicit = float(solver_parameters.get("theta_implicit", 1.0))
    use_predictor_corrector = bool(solver_parameters.get("use_predictor_corrector", True))
    n_corrector_steps = int(solver_parameters.get("n_corrector_steps", 1))
    save_n = solver_parameters.get("save_n", solver_parameters.get("n_save"))
    theta_linear_solver = str(solver_parameters.get("theta_linear_solver", "direct")).strip().lower()
    theta_gmres_tol = float(solver_parameters.get("theta_gmres_tol", 1.0e-8))
    theta_gmres_maxiter = int(solver_parameters.get("theta_gmres_maxiter", 200))
    radau_linear_solver = str(solver_parameters.get("radau_linear_solver", "direct")).strip().lower()
    radau_gmres_tol = float(solver_parameters.get("radau_gmres_tol", 1.0e-8))
    radau_gmres_maxiter = int(solver_parameters.get("radau_gmres_maxiter", 200))
    radau_error_estimator = str(solver_parameters.get("radau_error_estimator", "embedded2")).strip().lower()
    radau_newton_strategy = str(solver_parameters.get("radau_newton_strategy", "simplified")).strip().lower()
    radau_debug_print_er = bool(solver_parameters.get("debug_print_er", False))

    if family == "theta":
        tol = float(
            solver_parameters.get(
                "nonlinear_solver_tol",
                solver_parameters.get("residual_tol", 1e-8),
            )
        )
        maxiter = int(
            solver_parameters.get(
                "nonlinear_solver_maxiter",
                solver_parameters.get("n_max_iterations", 50),
            )
        )
        if backend == "theta_linear":
            return ThetaLinearSolver(
                t0=t0,
                t1=t1,
                dt=dt,
                theta_implicit=theta_implicit,
                use_predictor_corrector=use_predictor_corrector,
                n_corrector_steps=n_corrector_steps,
                save_n=save_n,
            )
        if backend == "theta_newton":
            return ThetaNewtonSolver(
                t0=t0,
                t1=t1,
                dt=dt,
                theta_implicit=theta_implicit,
                tol=tol,
                maxiter=maxiter,
                ptc_enabled=bool(solver_parameters.get("theta_ptc_enabled", True)),
                ptc_dt_min_factor=float(solver_parameters.get("theta_ptc_dt_min_factor", 1.0e-4)),
                ptc_dt_max_factor=float(solver_parameters.get("theta_ptc_dt_max_factor", 1.0e3)),
                ptc_growth=float(solver_parameters.get("theta_ptc_growth", 1.5)),
                ptc_shrink=float(solver_parameters.get("theta_ptc_shrink", 0.5)),
                line_search_enabled=bool(solver_parameters.get("theta_line_search_enabled", True)),
                line_search_contraction=float(solver_parameters.get("theta_line_search_contraction", 0.5)),
                line_search_min_alpha=float(solver_parameters.get("theta_line_search_min_alpha", 1.0e-4)),
                line_search_c=float(solver_parameters.get("theta_line_search_c", 1.0e-4)),
                max_step_retries=int(solver_parameters.get("theta_max_step_retries", 8)),
                linear_solver=theta_linear_solver,
                gmres_tol=theta_gmres_tol,
                gmres_maxiter=theta_gmres_maxiter,
                trust_region_enabled=bool(solver_parameters.get("theta_trust_region_enabled", False)),
                trust_radius=float(solver_parameters.get("theta_trust_radius", 1.0)),
                homotopy_steps=int(solver_parameters.get("theta_homotopy_steps", 1)),
                differentiable_mode=bool(solver_parameters.get("theta_differentiable_mode", False)),
                save_n=save_n,
            )
        raise ValueError(
            f"Unknown theta transport backend '{backend}'. "
            "Expected one of ['theta_linear', 'theta_newton']."
        )

    if family == "nonlinear":
        tol = float(
            solver_parameters.get(
                "nonlinear_solver_tol",
                solver_parameters.get("residual_tol", 1e-8),
            )
        )
        maxiter = int(
            solver_parameters.get(
                "nonlinear_solver_maxiter",
                solver_parameters.get("n_max_iterations", 50),
            )
        )
        if backend == "newton":
            return NewtonSolver(tol=tol, maxiter=maxiter)
        if backend == "anderson":
            m = int(getattr(solver_parameters, "anderson_history", 5))
            return AndersonSolver(m=m, tol=tol, maxiter=maxiter)
        if backend == "broyden":
            return BroydenSolver(tol=tol, maxiter=maxiter)
        if backend == "jaxopt_steady_state":
            return JaxoptSteadyStateSolver(tol=tol, maxiter=maxiter)
        raise ValueError(
            f"Unknown nonlinear transport backend '{backend}'. "
            "Expected one of ['newton', 'anderson', 'broyden', 'jaxopt_steady_state']."
        )

    if backend == "radau":
        return RADAUSolver(
            t0=t0,
            t1=t1,
            dt=dt,
            rtol=float(solver_parameters.get("rtol", 1.0e-6)),
            atol=float(solver_parameters.get("atol", 1.0e-8)),
            max_step=float(solver_parameters.get("max_step", max(t1 - t0, dt))),
            min_step=float(solver_parameters.get("min_step", 1.0e-14)),
            tol=float(
                solver_parameters.get(
                    "tol",
                    solver_parameters.get(
                    "nonlinear_solver_tol",
                    solver_parameters.get("residual_tol", 1.0e-8),
                    )
                )
            ),
            maxiter=int(
                solver_parameters.get(
                    "maxiter",
                    solver_parameters.get(
                    "nonlinear_solver_maxiter",
                    solver_parameters.get("n_max_iterations", 50),
                    )
                )
            ),
            linear_solver=radau_linear_solver,
            gmres_tol=radau_gmres_tol,
            gmres_maxiter=radau_gmres_maxiter,
            error_estimator=radau_error_estimator,
            newton_strategy=radau_newton_strategy,
            debug_print_er=radau_debug_print_er,
            safety_factor=float(solver_parameters.get("safety_factor", 0.9)),
            min_step_factor=float(solver_parameters.get("min_step_factor", 0.1)),
            max_step_factor=float(solver_parameters.get("max_step_factor", 5.0)),
            save_n=save_n,
        )

    if backend in ("predictor_corrector", "heun"):
        return PredictorCorrectorSolver(t0=t0, t1=t1, dt=dt, save_n=save_n)

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
        max_steps=None,
    )


# ============================================================================
# RADAU Method: 3-stage implicit RK with 5th-order accuracy
# ============================================================================

@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class RADAUSolver(TransportSolver):
    """
    5th-order RADAU method with 3 stages.
    Implicit Runge-Kutta designed for stiff ODEs with excellent stability.
    
    Stages: c = [1/5 - sqrt(6)/10, 1/5 + sqrt(6)/10, 1]
    Weights and nodes carefully chosen for stiff systems.
    
    Features:
      - Jitable and differentiable via JAX control flow
      - Adaptive timestep with error estimation
      - Newton iteration per stage with configurable linear solver
      - Dense output via Hermite interpolation (optional)
    """
    
    t0: float
    t1: float
    dt: float
    rtol: float = 1e-6
    atol: float = 1e-8
    max_step: float = 1.0
    min_step: float = 1.0e-14
    tol: float = 1.0e-8
    maxiter: int = 50
    linear_solver: str = "direct"
    gmres_tol: float = 1.0e-8
    gmres_maxiter: int = 200
    error_estimator: str = "embedded2"
    newton_strategy: str = "simplified"
    debug_print_er: bool = False
    safety_factor: float = 0.9
    min_step_factor: float = 0.1
    max_step_factor: float = 5.0
    n_steps: int = 0

    def __init__(
        self,
        t0: float = 0.0,
        t1: float = 1.0,
        dt: float = 0.1,
        rtol: float = 1e-6,
        atol: float = 1e-8,
        max_step: float = 1.0,
        min_step: float = 1.0e-14,
        tol: float = 1.0e-8,
        maxiter: int = 50,
        linear_solver: str = "direct",
        gmres_tol: float = 1.0e-8,
        gmres_maxiter: int = 200,
        error_estimator: str = "embedded2",
        newton_strategy: str = "simplified",
        debug_print_er: bool = False,
        safety_factor: float = 0.9,
        min_step_factor: float = 0.1,
        max_step_factor: float = 5.0,
        vector_field: Callable | None = None,
        save_n=None,
    ):
        """Initialize RADAU solver."""
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
        object.__setattr__(self, "debug_print_er", bool(debug_print_er))
        object.__setattr__(self, "safety_factor", float(safety_factor))
        object.__setattr__(self, "min_step_factor", float(min_step_factor))
        object.__setattr__(self, "max_step_factor", float(max_step_factor))
        object.__setattr__(self, "n_steps", n_steps)
        if vector_field is not None:
            object.__setattr__(self, "_vector_field", vector_field)
        object.__setattr__(self, "save_n", save_n)

    def solve(self, state, vector_field: Callable, *args, **kwargs):
        """Integrate from t0 to t1 using a 3-stage Radau IIA method with adaptive timestep control."""
        flat_state0, unravel = jax.flatten_util.ravel_pytree(state)
        state_dtype = flat_state0.dtype
        sqrt6 = jnp.sqrt(jnp.asarray(6.0, dtype=state_dtype))
        c = jnp.asarray([(4.0 - sqrt6) / 10.0, (4.0 + sqrt6) / 10.0, 1.0], dtype=state_dtype)
        a = jnp.asarray(
            [
                [(88.0 - 7.0 * sqrt6) / 360.0, (296.0 - 169.0 * sqrt6) / 1800.0, (-2.0 + 3.0 * sqrt6) / 225.0],
                [(296.0 + 169.0 * sqrt6) / 1800.0, (88.0 + 7.0 * sqrt6) / 360.0, (-2.0 - 3.0 * sqrt6) / 225.0],
                [(16.0 - sqrt6) / 36.0, (16.0 + sqrt6) / 36.0, 1.0 / 9.0],
            ],
            dtype=state_dtype,
        )
        b = a[2]
        # Low-memory embedded second-order estimator using the existing stage
        # derivatives. This avoids step-doubling, which was tripling implicit
        # stage solves and inflating compile-time memory.
        b_embedded = jnp.asarray(
            [
                0.5 - 0.5 / sqrt6,
                0.5 + 0.5 / sqrt6,
                0.0,
            ],
            dtype=state_dtype,
        )
        order = 5.0
        error_order = 2.0 if self.error_estimator == "embedded2" else order
        state_dim = flat_state0.shape[0]
        t_final = jnp.asarray(self.t1, dtype=state_dtype)
        dt_min = jnp.asarray(self.min_step, dtype=state_dtype)
        dt_max = jnp.asarray(self.max_step, dtype=state_dtype)
        base_dt = jnp.clip(jnp.asarray(self.dt, dtype=state_dtype), dt_min, dt_max)
        max_total_steps = int(max(8, self.n_steps * 16))

        def _flatten_rhs(t_value, flat_y):
            rhs_tree = vector_field(t_value, unravel(flat_y), *args, **kwargs)
            rhs_flat, _ = jax.flatten_util.ravel_pytree(rhs_tree)
            return rhs_flat

        def _single_step(flat_y, t_value, h_value):
            f0 = _flatten_rhs(t_value, flat_y)
            z0 = jnp.tile(f0, 3)

            def residual_with_stages(z_flat):
                stages = z_flat.reshape((3, state_dim))
                stage_states = flat_y[None, :] + h_value * (a @ stages)
                evals = jnp.stack(
                    [
                        _flatten_rhs(t_value + c[i] * h_value, stage_states[i])
                        for i in range(3)
                    ],
                    axis=0,
                )
                residual_flat = (stages - evals).reshape((-1,))
                return residual_flat, stage_states

            def residual(z_flat):
                residual_flat, _ = residual_with_stages(z_flat)
                return residual_flat

            if self.newton_strategy == "simplified":
                _, stage_states_ref = residual_with_stages(z0)
                if self.linear_solver == "gmres":
                    stage_linears_ref = tuple(
                        jax.linearize(
                            lambda y_stage, t_stage=t_value + c[i] * h_value: _flatten_rhs(t_stage, y_stage),
                            stage_states_ref[i],
                        )[1]
                        for i in range(3)
                    )
                    stage_solver = lambda rhs: jax.scipy.sparse.linalg.gmres(
                        lambda v_flat: _radau_stage_matvec(v_flat, stage_linears_ref, a, h_value, state_dim),
                        rhs,
                        tol=self.gmres_tol,
                        atol=self.gmres_tol,
                        maxiter=self.gmres_maxiter,
                    )[0]
                else:
                    jac_ref = jax.jacfwd(residual)(z0)
                    stage_solver = lambda rhs: jax.scipy.linalg.solve(jac_ref, rhs)
            else:
                stage_solver = None

            def body_fn(newton_state):
                iter_idx, z_cur, delta_norm, residual_norm = newton_state
                residual_cur, stage_states = residual_with_stages(z_cur)
                if self.newton_strategy == "simplified":
                    delta = stage_solver(-residual_cur)
                else:
                    if self.linear_solver == "gmres":
                        stage_linears = tuple(
                            jax.linearize(
                                lambda y_stage, t_stage=t_value + c[i] * h_value: _flatten_rhs(t_stage, y_stage),
                                stage_states[i],
                            )[1]
                            for i in range(3)
                        )
                        delta, _ = jax.scipy.sparse.linalg.gmres(
                            lambda v_flat: _radau_stage_matvec(v_flat, stage_linears, a, h_value, state_dim),
                            -residual_cur,
                            tol=self.gmres_tol,
                            atol=self.gmres_tol,
                            maxiter=self.gmres_maxiter,
                        )
                    else:
                        jac = jax.jacfwd(residual)(z_cur)
                        delta = jax.scipy.linalg.solve(jac, -residual_cur)
                delta = jnp.where(jnp.all(jnp.isfinite(delta)), delta, jnp.zeros_like(delta))
                z_next = z_cur + delta
                return (
                    iter_idx + 1,
                    z_next,
                    jnp.linalg.norm(delta),
                    jnp.linalg.norm(residual_cur),
                )

            def cond_fn(newton_state):
                iter_idx, _, delta_norm, residual_norm = newton_state
                active = jnp.logical_and(
                    residual_norm > self.tol,
                    delta_norm > self.tol,
                )
                return jnp.logical_and(iter_idx < self.maxiter, active)

            init_newton = (
                jnp.asarray(0),
                z0,
                jnp.asarray(jnp.inf, dtype=state_dtype),
                jnp.asarray(jnp.inf, dtype=state_dtype),
            )
            _, z_final, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_newton)
            stages_final = z_final.reshape((3, state_dim))
            final_residual = residual(z_final)
            converged = jnp.logical_and(
                jnp.all(jnp.isfinite(z_final)),
                jnp.linalg.norm(final_residual) <= self.tol,
            )
            flat_next = flat_y + h_value * (b @ stages_final)
            flat_next_embedded = flat_y + h_value * (b_embedded @ stages_final)
            return flat_next, flat_next_embedded, converged

        def _scaled_error(candidate, reference):
            scale = self.atol + self.rtol * jnp.maximum(jnp.abs(candidate), jnp.abs(reference))
            normalized = (candidate - reference) / scale
            return jnp.sqrt(jnp.mean(normalized * normalized) + 1.0e-30)

        def _attempt_step(carry):
            t_value, flat_y, dt_value, done, failed, fail_code, n_accepted = carry
            trial_dt0 = jnp.minimum(dt_value, t_final - t_value)
            init_retry = (
                jnp.asarray(0),
                trial_dt0,
                flat_y,
                jnp.asarray(False),
                jnp.asarray(False),
                trial_dt0,
                trial_dt0,
            )

            def cond_fn(retry_state):
                retry_count, trial_dt, _, accepted, converged, _, _ = retry_state
                need_retry = jnp.logical_not(jnp.logical_and(accepted, converged))
                return jnp.logical_and(need_retry, jnp.logical_and(retry_count < self.maxiter, trial_dt > dt_min))

            def body_fn(retry_state):
                retry_count, trial_dt, _, _, _, _, _ = retry_state
                high_step, low_step, converged = _single_step(flat_y, t_value, trial_dt)
                error_norm = _scaled_error(high_step, low_step)
                accepted = jnp.logical_and(converged, error_norm <= 1.0)
                safe_error = jnp.maximum(error_norm, 1.0e-12)
                growth = self.safety_factor * safe_error ** (-1.0 / (error_order + 1.0))
                growth = jnp.clip(growth, self.min_step_factor, self.max_step_factor)
                next_dt = jnp.clip(trial_dt * growth, dt_min, dt_max)

                def accept_state(_):
                    return retry_count + 1, trial_dt, high_step, accepted, converged, trial_dt, next_dt

                def reject_state(_):
                    reduced_dt = jnp.clip(next_dt, dt_min, trial_dt * self.safety_factor)
                    return retry_count + 1, reduced_dt, flat_y, accepted, converged, reduced_dt, reduced_dt

                return jax.lax.cond(accepted, accept_state, reject_state, operand=None)

            _, retry_dt, retry_y, accepted, converged, accepted_dt, next_dt = jax.lax.while_loop(cond_fn, body_fn, init_retry)

            def _accept(_):
                dt_used = accepted_dt
                t_new = t_value + dt_used
                done_new = t_new >= (t_final - 1.0e-15)
                if self.debug_print_er:
                    accepted_state = unravel(retry_y)
                    if hasattr(accepted_state, "Er"):
                        jax.debug.print(
                            "RADAU accepted step: t={t}, dt={dt}, Er={er}",
                            t=t_new,
                            dt=dt_used,
                            er=accepted_state.Er,
                        )
                return (
                    t_new,
                    retry_y,
                    next_dt,
                    done_new,
                    jnp.asarray(False),
                    fail_code,
                    n_accepted + 1,
                ), (
                    retry_y,
                    t_new,
                    dt_used,
                    jnp.asarray(True),
                    jnp.asarray(False),
                    fail_code,
                )

            def _reject(_):
                code = jnp.where(converged, 2, 1)
                reduced_dt = jnp.maximum(jnp.minimum(retry_dt, dt_value * 0.5), dt_min)
                return (
                    t_value,
                    flat_y,
                    reduced_dt,
                    done,
                    jnp.asarray(True),
                    code,
                    n_accepted,
                ), (
                    flat_y,
                    t_value,
                    jnp.asarray(0.0, dtype=state_dtype),
                    jnp.asarray(False),
                    jnp.asarray(True),
                    code,
                )

            return jax.lax.cond(accepted, _accept, _reject, operand=None)

        def step_fn(carry, _):
            t_value, flat_y, dt_value, done, failed, fail_code, n_accepted = carry

            def _skip(_):
                return carry, (
                    flat_y,
                    t_value,
                    jnp.asarray(0.0, dtype=state_dtype),
                    jnp.asarray(False),
                    failed,
                    fail_code,
                )

            def _run(_):
                return _attempt_step(carry)

            return jax.lax.cond(jnp.logical_or(done, failed), _skip, _run, operand=None)

        init_carry = (
            jnp.asarray(self.t0, dtype=jnp.float64),
            flat_state0,
            base_dt,
            jnp.asarray(False),
            jnp.asarray(False),
            jnp.asarray(0),
            jnp.asarray(0),
        )

        save_n = getattr(self, 'save_n', None)
        # --- PATCH: Enforce quasi-neutrality after each step and on final state ---
        from ._transport_equations import enforce_quasi_neutrality
        species = _extract_species_from_args(args)

        # Remove all calls to enforce_quasi_neutrality inside stepping logic
        # Only apply to output states after all stepping is complete

        if save_n is not None and save_n > 1:
            save_times = jnp.linspace(self.t0, self.t1, save_n)
            y0, t0, dt0, acc0, fail0, code0 = step_fn(init_carry, None)[1]
            ys_saved = jnp.zeros((save_n,) + y0.shape, dtype=y0.dtype)
            ts_saved = jnp.zeros((save_n,), dtype=t0.dtype)
            dts_saved = jnp.zeros((save_n,), dtype=dt0.dtype)
            accepted_mask_saved = jnp.zeros((save_n,), dtype=acc0.dtype)
            failed_mask_saved = jnp.zeros((save_n,), dtype=fail0.dtype)
            fail_codes_saved = jnp.zeros((save_n,), dtype=code0.dtype)
            def cond_fun(loop_carry):
                t, y, dt_current, done, failed, fail_code, n_acc, save_idx, ys, ts, dts, accs, fails, codes = loop_carry
                return jnp.logical_and(save_idx < save_n, jnp.logical_not(done))
            def body_fun(loop_carry):
                t, y, dt_current, done, failed, fail_code, n_acc, save_idx, ys, ts, dts, accs, fails, codes = loop_carry
                (t_new, y_new, dt_next, done_new, failed_new, fail_code_new, n_acc_new), (y_out, t_out, dt_out, acc_out, fail_out, code_out) = step_fn((t, y, dt_current, done, failed, fail_code, n_acc), None)
                save_cond = jnp.logical_and(t_new >= save_times[save_idx], save_idx < save_n)
                ys = ys.at[save_idx].set(jax.lax.select(save_cond, y_new, ys[save_idx]))
                ts = ts.at[save_idx].set(jax.lax.select(save_cond, t_new, ts[save_idx]))
                dts = dts.at[save_idx].set(jax.lax.select(save_cond, dt_next, dts[save_idx]))
                accs = accs.at[save_idx].set(jax.lax.select(save_cond, acc_out, accs[save_idx]))
                fails = fails.at[save_idx].set(jax.lax.select(save_cond, fail_out, fails[save_idx]))
                codes = codes.at[save_idx].set(jax.lax.select(save_cond, code_out, codes[save_idx]))
                save_idx_new = save_idx + save_cond.astype(int)
                return (t_new, y_new, dt_next, done_new, failed_new, fail_code_new, n_acc_new, save_idx_new, ys, ts, dts, accs, fails, codes)
            loop_carry = (self.t0, y0, base_dt, False, False, 0, 0, 0, ys_saved, ts_saved, dts_saved, accepted_mask_saved, failed_mask_saved, fail_codes_saved)
            loop_carry = jax.lax.while_loop(cond_fun, body_fun, loop_carry)
            t_f, y_f_flat, _, done_f, failed_f, fail_code_f, n_acc_f, _, ys_saved, ts_saved, dts_saved, accepted_mask_saved, failed_mask_saved, fail_codes_saved = loop_carry
            ys_saved = jax.vmap(unravel)(ys_saved)
        else:
            def cond_fun(loop_carry):
                t_value, flat_y, dt_value, done, failed, fail_code, n_accepted, step_idx = loop_carry
                active = jnp.logical_not(jnp.logical_or(done, failed))
                return jnp.logical_and(step_idx < max_total_steps, active)

            def body_fun(loop_carry):
                t_value, flat_y, dt_value, done, failed, fail_code, n_accepted, step_idx = loop_carry
                (t_new, y_new, dt_next, done_new, failed_new, fail_code_new, n_acc_new), _ = step_fn(
                    (t_value, flat_y, dt_value, done, failed, fail_code, n_accepted),
                    None,
                )
                return (t_new, y_new, dt_next, done_new, failed_new, fail_code_new, n_acc_new, step_idx + 1)

            loop_carry = (
                jnp.asarray(self.t0, dtype=state_dtype),
                flat_state0,
                base_dt,
                jnp.asarray(False),
                jnp.asarray(False),
                jnp.asarray(0),
                jnp.asarray(0),
                jnp.asarray(0),
            )
            t_f, y_f_flat, dt_last, done_f, failed_f, fail_code_f, n_acc_f, _ = jax.lax.while_loop(cond_fun, body_fun, loop_carry)
            ys_saved = _save_state_series(unravel(y_f_flat))
            ts_saved = _save_scalar_series(t_f)
            dts_saved = _save_scalar_series(dt_last)
            accepted_mask_saved = _save_scalar_series(jnp.logical_not(failed_f))
            failed_mask_saved = _save_scalar_series(failed_f)
            fail_codes_saved = _save_scalar_series(fail_code_f)
        # Only enforce quasi-neutrality on output states
        if species is not None:
            ys_saved = jax.vmap(lambda s: enforce_quasi_neutrality(s, species))(ys_saved)
        final_state = unravel(y_f_flat)
        if species is not None:
            final_state = enforce_quasi_neutrality(final_state, species)
        return {
            "ys": ys_saved,
            "ts": ts_saved,
            "dts": dts_saved,
            "accepted_mask": accepted_mask_saved,
            "failed_mask": failed_mask_saved,
            "fail_codes": fail_codes_saved,
            "n_steps": n_acc_f,
            "done": done_f,
            "failed": failed_f,
            "fail_code": fail_code_f,
            "final_state": final_state,
            "final_time": t_f,
            "t0": self.t0,
            "t1": self.t1,
        }


register_time_solver("diffrax_kvaerno5", lambda **kw: DiffraxSolver(_get_diffrax_integrator("diffrax_kvaerno5"), **kw))
register_time_solver("diffrax_tsit5", lambda **kw: DiffraxSolver(_get_diffrax_integrator("diffrax_tsit5"), **kw))
register_time_solver("diffrax_dopri5", lambda **kw: DiffraxSolver(_get_diffrax_integrator("diffrax_dopri5"), **kw))
register_time_solver("radau", lambda **kw: RADAUSolver(**kw))
register_time_solver("predictor_corrector", lambda **kw: PredictorCorrectorSolver(**kw))
register_time_solver("heun", lambda **kw: PredictorCorrectorSolver(**kw))
register_time_solver("newton", lambda **kw: NewtonSolver(**kw))
register_time_solver("anderson", lambda **kw: AndersonSolver(**kw))
register_time_solver("broyden", lambda **kw: BroydenSolver(**kw))
register_time_solver("jaxopt_steady_state", lambda **kw: JaxoptSteadyStateSolver(**kw))
register_time_solver("theta_linear", lambda **kw: ThetaLinearSolver(**kw))
register_time_solver("theta_newton", lambda **kw: ThetaNewtonSolver(**kw))
