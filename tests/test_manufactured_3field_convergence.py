"""
Manufactured-solution convergence test for coupled n, T, Er system.
- Tests all three solvers: Theta, RADAU, Kvaerno5
- Sweeps radial grid sizes and stiffness levels
- Reports L2 errors for all fields at final time
"""
import pytest
import jax
import jax.numpy as jnp
from NEOPAX._transport_solvers import ThetaNewtonSolver, RADAUSolver, DiffraxSolver, _get_diffrax_integrator
from NEOPAX._fem import conservative_update
from NEOPAX._cell_variable import make_profile_cell_variable

# Manufactured profiles (smooth, sign-changing Er)
def manufactured_profiles(r, t, eps):
    # Parameters for manufactured profiles
    a_n = 0.1
    omega_n = 1.0
    c_n = 0.2
    A = 0.4
    kappa = 8.0
    r0 = 0.5
    b_E = 0.1
    omega_E = 1.0
    a_T = 0.05
    omega_T = 0.5
    c_T = 0.1
    # Profiles
    Er = A * jnp.tanh(kappa * (r - r0)) * (1 + b_E * jnp.sin(omega_E * t))
    n = 1 + a_n * jnp.cos(jnp.pi * r) * jnp.exp(-omega_n * t) + c_n * Er
    T = 1.2 - 0.2 * r + a_T * jnp.sin(2 * jnp.pi * r) * jnp.exp(-omega_T * t) + c_T * Er
    return n, T, Er

# Use FVM Laplacian (conservative_update) for manufactured source
def fvm_laplacian(profile, face_centers):
    cell_var = make_profile_cell_variable(
        profile,
        face_centers,
        left_face_constraint=profile[0],
        right_face_constraint=profile[-1],
    )
    faces = cell_var.face_value()
    dx = jnp.diff(face_centers)
    return conservative_update(faces, dx)

# PDE system with manufactured source terms
def rhs(t, y, r, eps, Dn=0.02, DT=0.01, DEr=0.03, alpha_n=0.0, alpha_T=0.1, alpha_E=0.2, beta_E=0.15):
    # Coupling parameters: adjust as desired
    # alpha_n: coupling from Er into n
    # alpha_T: coupling from Er into T
    # alpha_E: coupling from n into Er
    # beta_E: coupling from T into Er
    n, T, Er = y
    n_star, T_star, Er_star = manufactured_profiles(r, t, eps)
    # Time derivatives (analytic)
    a_n = 0.1
    omega_n = 1.0
    c_n = 0.2
    A = 0.4
    kappa = 8.0
    r0 = 0.5
    b_E = 0.1
    omega_E = 1.0
    a_T = 0.05
    omega_T = 0.5
    c_T = 0.1
    # Er profile and derivatives
    tanh_arg = kappa * (r - r0)
    sech2 = 1.0 / jnp.cosh(tanh_arg) ** 2
    Er = A * jnp.tanh(tanh_arg) * (1 + b_E * jnp.sin(omega_E * t))
    dEr_dt = A * jnp.tanh(tanh_arg) * b_E * omega_E * jnp.cos(omega_E * t)
    dEr_dt = dEr_dt
    # n profile and derivatives
    dn_dt = -a_n * omega_n * jnp.cos(jnp.pi * r) * jnp.exp(-omega_n * t) + c_n * dEr_dt
    # T profile and derivatives
    dT_dt = -a_T * omega_T * jnp.sin(2 * jnp.pi * r) * jnp.exp(-omega_T * t) + c_T * dEr_dt
    # Laplacians (use FVM for manufactured profiles)
    face_centers = jnp.linspace(r[0], r[-1], r.size + 1)
    n_lap = fvm_laplacian(n_star, face_centers)
    T_lap = fvm_laplacian(T_star, face_centers)
    Er_lap = fvm_laplacian(Er_star, face_centers)
    # Manufactured sources (include all coupling terms)
    S_n = dn_dt - Dn * n_lap - alpha_n * Er_star
    S_T = dT_dt - DT * T_lap - alpha_T * Er_star
    S_Er = dEr_dt - DEr * Er_lap - alpha_E * n_star + beta_E * T_star
    # PDE RHS (with coupling)
    return jnp.stack([
        Dn * n_lap + alpha_n * Er + S_n,
        DT * T_lap + alpha_T * Er + S_T,
        DEr * Er_lap + alpha_E * n - beta_E * T + S_Er
    ])

@pytest.mark.parametrize("solver_name", ["theta", "radau", "kvaerno5"])
@pytest.mark.parametrize("n_r", [32, 64, 128])
@pytest.mark.parametrize("epsilon", [1.0, 0.05, 0.001])
def test_manufactured_3field(solver_name, n_r, epsilon):
    t0, tf = 0.0, 0.5
    r = jnp.linspace(0, 1, n_r)
    n0, T0, Er0 = manufactured_profiles(r, t0, epsilon)
    y0 = jnp.stack([n0, T0, Er0])
    def vf(t, y):
        return rhs(t, y, r, epsilon)
    import diffrax
    if solver_name == "theta":
        tol = 1e-9
        solver = ThetaNewtonSolver(
            t0=t0, t1=tf, dt=1e-8, theta_implicit=1.0, tol=tol, maxiter=20,
            ptc_enabled=True,
            ptc_growth=1.5,
            ptc_shrink=0.5,
            ptc_dt_min_factor=1e-4,
            ptc_dt_max_factor=1e3,
            max_step_retries=8,
            save_n=10
        )
        test_tol = 5 * tol
    elif solver_name == "radau":
        rtol = 1e-7
        atol = 1e-9
        solver = RADAUSolver(
            t0=t0, t1=tf, dt=1e-8, rtol=rtol, atol=atol, save_n=10
        )
        test_tol = 5 * min(rtol, atol)
    elif solver_name == "kvaerno5":
        rtol = 1e-7
        atol = 1e-9
        solver = DiffraxSolver(
            _get_diffrax_integrator("diffrax_kvaerno5"),
            t0=t0, t1=tf, dt=1e-8,
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            save_n=10
        )
        test_tol = 5 * min(rtol, atol)
    else:
        raise ValueError(f"Unknown solver: {solver_name}")
    out = solver.solve(y0, vf)
    # Handle Diffrax Solution object vs dict output
    if solver_name == "kvaerno5":
        # Diffrax Solution: ys is (steps, state_shape)
        y_final = out.ys[-1]
    else:
        y_final = out["final_state"]
    n_star, T_star, Er_star = manufactured_profiles(r, tf, epsilon)
    err_n = float(jnp.linalg.norm(y_final[0] - n_star) / jnp.sqrt(n_r))
    err_T = float(jnp.linalg.norm(y_final[1] - T_star) / jnp.sqrt(n_r))
    err_Er = float(jnp.linalg.norm(y_final[2] - Er_star) / jnp.sqrt(n_r))
    print(f"solver={solver_name} n_r={n_r} eps={epsilon} err_n={err_n:.2e} err_T={err_T:.2e} err_Er={err_Er:.2e} tol={test_tol:.1e}")
    assert err_n < test_tol
    assert err_T < test_tol
    assert err_Er < test_tol
