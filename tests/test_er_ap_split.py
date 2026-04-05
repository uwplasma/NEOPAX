import dataclasses

import jax
import jax.numpy as jnp

from NEOPAX._parameters import Solver_Parameters
from NEOPAX._state import TransportState
from NEOPAX._transport_solvers import RADAUSolver, ThetaNewtonSolver
from NEOPAX._transport_equations import ElectricFieldEquation


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class MockField:
    r_grid: jnp.ndarray
    r_grid_half: jnp.ndarray
    Vprime: jnp.ndarray
    Vprime_half: jnp.ndarray
    enlogation: jnp.ndarray
    iota: jnp.ndarray
    B0: jnp.ndarray


def _make_mock_field(n_r: int):
    r_half = jnp.linspace(0.0, 1.0, n_r + 1)
    r = 0.5 * (r_half[1:] + r_half[:-1])
    return MockField(
        r_grid=r,
        r_grid_half=r_half,
        Vprime=jnp.ones(n_r),
        Vprime_half=jnp.ones(n_r + 1),
        enlogation=jnp.ones(n_r),
        iota=jnp.ones(n_r),
        B0=jnp.ones(n_r),
    )


def test_ap_preconditioner_parameter_default_false():
    p = Solver_Parameters()
    assert p.use_ap_er_preconditioner is False


def test_er_ap_linear_split_shapes_and_identity():
    n_r = 8
    field = _make_mock_field(n_r)

    state = TransportState(
        density=jnp.ones((1, n_r)),
        temperature=jnp.ones((1, n_r)),
        Er=jnp.linspace(-0.2, 0.2, n_r),
    )

    flux_models = {"Gamma_total": jnp.ones((1, n_r)) * 0.01}
    params = Solver_Parameters(DEr=1.0, Er_relax=0.2, use_ap_er_preconditioner=True)

    eq = ElectricFieldEquation()
    diag, src = eq.ap_linear_split(
        state,
        flux_models,
        None,
        field,
        params,
        charge_qp=jnp.array([1.0]),
        species_mass=jnp.array([1.0]),
    )

    rhs = eq(
        state,
        flux_models,
        None,
        field,
        params,
        charge_qp=jnp.array([1.0]),
        species_mass=jnp.array([1.0]),
    )

    assert diag.shape == (n_r,)
    assert src.shape == (n_r,)
    assert jnp.all(jnp.isfinite(diag))
    assert jnp.all(jnp.isfinite(src))
    assert jnp.allclose(diag * state.Er + src, rhs, rtol=1e-7, atol=1e-9)


def test_theta_solver_accepts_ap_preconditioner_hook():
    solver = ThetaNewtonSolver(
        t0=0.0,
        t1=0.2,
        dt=1.0e-2,
        theta_implicit=1.0,
        tol=1.0e-9,
        maxiter=20,
        ptc_enabled=True,
        line_search_enabled=True,
        max_step_retries=8,
        differentiable_mode=False,
    )

    y0 = jnp.array([1.0])

    def vf(t, y):
        del t
        return -10.0 * y

    def ap_diag(t, y):
        del t, y
        return jnp.array([5.0])

    out = solver.solve(y0, vf, ap_preconditioner=ap_diag)
    assert jnp.all(jnp.isfinite(out["final_state"]))


class _StiffDampedErToyModel:
    """Fast-slow toy model with a large *stable* Er damping.

    dEr/dt = (-Er + source(n, T)) / epsilon  — stiff but unconditionally stable.

    The AP diagonal for this model is the exact stiff Jacobian entry
    df_Er/dEr = -1/epsilon (negative, hence the residual Jacobian diagonal
    1 + dt/epsilon stays large and positive).  Adding the AP contribution
    simply doubles the diagonal dominance without changing the Newton direction
    qualitatively, so both AP-off and AP-on converge cleanly.
    """

    def __init__(self, epsilon=0.01):
        self.epsilon = epsilon
        self.n_star = 1.0
        self.t_star = 1.0

    def __call__(self, t, y):
        del t
        n, T, Er = y
        dn = -(n - self.n_star) + 0.3 * Er
        dT = -(T - self.t_star) - 0.2 * Er
        dEr = (-Er + 0.5 * jnp.tanh(n - T)) / self.epsilon
        return jnp.array([dn, dT, dEr])


def _stable_ap_diag(problem, dt=2.5e-2):
    """AP diagonal for the stable-damped model.

    df_Er/dEr = -1/epsilon  (stiff stable).
    Residual Jacobian diagonal for theta=1: 1 + dt/epsilon.
    AP contribution: dt/epsilon — doubles the diagonal, strong diagonal dominance.
    """
    def _diag(t, y):
        del t, y
        return jnp.array([0.0, 0.0, dt / problem.epsilon])

    return _diag


def test_theta_ap_preconditioner_not_worse_than_off_on_damped_case():
    """AP hook invocation must not regress accuracy on a stiff stable Er problem.

    Uses a stiff-damped Er model (not bifurcating) so that both AP-off and AP-on
    converge cleanly via implicit theta.  The test verifies:
      1. Neither run fails.
      2. AP-on final-state error vs RADAU is not materially worse than AP-off.
    """
    problem = _StiffDampedErToyModel(epsilon=0.01)
    y0 = jnp.array([1.0, 1.0, 0.0])
    t0, tf = 0.0, 2.0

    theta_kwargs = dict(
        t0=t0,
        t1=tf,
        dt=2.5e-2,
        theta_implicit=1.0,
        tol=1.0e-9,
        maxiter=25,
        ptc_enabled=True,
        line_search_enabled=True,
        max_step_retries=12,
        differentiable_mode=False,
    )

    out_off = ThetaNewtonSolver(**theta_kwargs).solve(y0, problem)
    out_on = ThetaNewtonSolver(**theta_kwargs).solve(
        y0, problem, ap_preconditioner=_stable_ap_diag(problem)
    )
    ref = RADAUSolver(t0=t0, t1=tf, dt=2.0e-1, rtol=1.0e-7, atol=1.0e-9).solve(y0, problem)

    assert not bool(out_off["failed"]), "AP-off theta run failed unexpectedly"
    assert not bool(out_on["failed"]), "AP-on theta run failed — AP hook broke convergence"

    y_ref = ref["final_state"]
    err_off = float(jnp.linalg.norm(out_off["final_state"] - y_ref))
    err_on = float(jnp.linalg.norm(out_on["final_state"] - y_ref))

    # AP-on must not regress materially versus AP-off on this stiff stable model.
    assert err_on <= err_off * 1.25 + 5.0e-4, (
        f"AP-on error {err_on:.2e} >> AP-off error {err_off:.2e}"
    )


class _DiffusiveAmbipolarToyModel:
    """Semidiscrete Er toy with diffusion plus a multi-root ambipolar term.

    This mirrors the structure of the production equation more closely:

      dEr/dt = D * Laplacian(Er) - (Er^3 - lam * Er - drive) / epsilon

    The AP hook intentionally linearizes only the diffusion block, matching
    ElectricFieldEquation.ap_linear_split(). The ambipolar cubic stays fully
    explicit/nonlinear, so this test checks robustness on a chosen branch rather
    than claiming global convergence improvement across root transitions.
    """

    def __init__(self, epsilon=0.05, diffusion=0.02, lam=1.1):
        self.epsilon = epsilon
        self.diffusion = diffusion
        self.lam = lam
        self.drive = jnp.array([-0.24, -0.18, -0.12])
        self.inv_dx2 = 16.0

    def laplacian(self, er):
        left = er[0]
        right = er[-1]
        ext = jnp.concatenate([jnp.array([left]), er, jnp.array([right])])
        return self.inv_dx2 * (ext[:-2] - 2.0 * ext[1:-1] + ext[2:])

    def __call__(self, t, y):
        del t
        er = y
        diffusion_term = self.diffusion * self.laplacian(er)
        ambipolar_term = (er**3 - self.lam * er - self.drive) / self.epsilon
        return diffusion_term - ambipolar_term


def _diffusion_only_ap_diag(problem):
    diag = problem.diffusion * problem.inv_dx2 * jnp.array([1.0, 2.0, 1.0])

    def _diag(t, y):
        del t, y
        return diag

    return _diag


def test_theta_ap_preconditioner_preserves_branch_on_diffusive_ambipolar_toy():
    """AP-on stays on the same ambipolar branch for a diffusion-plus-source toy.

    This is the physics-shaped regression: diffusion is the only AP-linearized
    contribution, while the nonlinear ambipolar source keeps its multi-root
    structure. The assertion is intentionally modest: both runs should converge
    to the same branch as the high-order reference when initialized on that
    branch.
    """
    problem = _DiffusiveAmbipolarToyModel()
    y0 = jnp.array([-0.85, -0.75, -0.65])
    t0, tf = 0.0, 1.5

    theta_kwargs = dict(
        t0=t0,
        t1=tf,
        dt=1.0e-2,
        theta_implicit=1.0,
        tol=1.0e-9,
        maxiter=25,
        ptc_enabled=True,
        line_search_enabled=True,
        max_step_retries=12,
        differentiable_mode=False,
    )

    out_off = ThetaNewtonSolver(**theta_kwargs).solve(y0, problem)
    out_on = ThetaNewtonSolver(**theta_kwargs).solve(
        y0,
        problem,
        ap_preconditioner=_diffusion_only_ap_diag(problem),
    )
    ref = RADAUSolver(t0=t0, t1=tf, dt=5.0e-2, rtol=1.0e-7, atol=1.0e-9).solve(y0, problem)

    assert not bool(out_off["failed"]), "AP-off theta run failed on diffusive ambipolar toy"
    assert not bool(out_on["failed"]), "AP-on theta run failed on diffusive ambipolar toy"
    assert jnp.all(jnp.isfinite(out_off["final_state"]))
    assert jnp.all(jnp.isfinite(out_on["final_state"]))

    mean_ref = float(jnp.mean(ref["final_state"]))
    mean_off = float(jnp.mean(out_off["final_state"]))
    mean_on = float(jnp.mean(out_on["final_state"]))

    # The initial condition is on the negative-root branch; both theta runs
    # should stay on that branch and track the reference branch sign.
    assert mean_ref < 0.0
    assert mean_off < 0.0
    assert mean_on < 0.0

    err_off = float(jnp.linalg.norm(out_off["final_state"] - ref["final_state"]))
    err_on = float(jnp.linalg.norm(out_on["final_state"] - ref["final_state"]))

    assert err_off < 7.5e-2
    assert err_on < 1.2e-1
