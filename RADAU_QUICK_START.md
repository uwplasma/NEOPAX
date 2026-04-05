# Quick Start: RADAU Solver & Theta Benchmarks

## TL;DR

Two major additions to NEOPAX:

1. **RADAU5 Solver**: 5th-order implicit RK for stiff ODEs, full JAX support (JIT + autodiff)
2. **Comprehensive Benchmark Suite**: 5 analytical + 1 branchy problem, tests for theta/RADAU solvers

---

## 1. Using RADAU Solver

### Basic Integration

```python
import jax.numpy as jnp
from NEOPAX._transport_solvers import RADAUSolver

# Define your ODE: y' = f(t, y)
def f(t, y):
    return -100.0 * y  # Stiff decay example

# Create solver
solver = RADAUSolver(
    t0=0.0,
    t1=1.0,
    dt=0.01,
    rtol=1e-6,
    atol=1e-8,
)

# Solve
y0 = jnp.array([1.0])
result = solver.solve(y0, f)

print(f"Final value: {result['final_state']}")
print(f"Final time: {result['final_time']}")
```

### Via build_time_solver (Config-Based)

```python
from NEOPAX._parameters import Solver_Parameters
from NEOPAX._transport_solvers import build_time_solver

# Create parameters
params = Solver_Parameters()
params.transport_solver_backend = "radau"  # ← Select RADAU
params.t0 = 0.0
params.t_final = 1.0
params.dt = 0.01
params.rtol = 1e-6
params.atol = 1e-8

# Build solver
solver = build_time_solver(params)

# Use it
result = solver.solve(y0, f)
```

### With Custom Parameters

```python
radau = RADAUSolver(
    t0=0.0, t1=10.0, dt=0.1,
    rtol=1e-6, atol=1e-8,
    max_step=1.0,          # Limit step size
    max_step_factor=3.0,   # More aggressive growth
    maxiter=100,           # Better convergence
)

# For 2D or higher-dimensional problems:
y0 = jnp.array([1.0, 0.5, -0.5])  # 3D initial condition
result = radau.solve(y0, f_3d)
```

---

## 2. Running Benchmark Tests

### Install Dependencies

```bash
pip install pytest jax jaxtyping
```

### Run All Tests

```bash
cd NEOPAX
pytest tests/test_theta_solver_benchmarks.py -v
```

### Run Specific Equation Tests

```bash
# Only scalar decay (simplest)
pytest tests/test_theta_solver_benchmarks.py::TestScalarDecay -v

# Only 2x2 stiff linear (hardest)
pytest tests/test_theta_solver_benchmarks.py::TestStiff2x2Linear -v

# Only 3D branchy (stress test)
pytest tests/test_theta_solver_benchmarks.py::TestBranchyStiffToyModel -v
```

### Run Specific Solver Tests

```bash
# Test RADAU on all equations
pytest tests/test_theta_solver_benchmarks.py -k "radau" -v

# Test theta robust mode
pytest tests/test_theta_solver_benchmarks.py -k "theta_robust" -v

# Test theta smooth/differentiable mode
pytest tests/test_theta_solver_benchmarks.py -k "theta_smooth" -v
```

### Run Convergence & Differentiability

```bash
# Check global error convergence order
pytest tests/test_theta_solver_benchmarks.py::TestConvergence -v

# Check JIT and autodiff compatibility
pytest tests/test_theta_solver_benchmarks.py::TestDifferentiability -v
```

---

## 3. Understanding Benchmark Equations

| Name | Order | Analytical? | Stiffness | Purpose |
|------|-------|---|---|---|
| **Scalar Decay** | 1D | ✓ | High | Baseline test |
| **Forced Linear** | 1D | ✓ | Mid | Transient + periodic response |
| **2×2 Stiff Linear** | 2D | ✓ | Very High | Extreme stiffness |
| **Logistic** | 1D | ✓ | None | Smooth nonlinearity |
| **Branchy 3D** | 3D | ✗ | Mixed | Stress test, branchy transients |

---

## 4. Example: Comparing Solvers

```python
import jax.numpy as jnp
from NEOPAX._transport_solvers import ThetaNewtonSolver, RADAUSolver

# Define stiff problem
def f(t, y):
    return -100.0 * y + 10.0 * jnp.sin(t)

y0 = jnp.array([0.0])

# Theta robust
theta = ThetaNewtonSolver(
    t0=0.0, t1=10.0, dt=0.1,
    theta_implicit=1.0,
    theta_ptc_enabled=True,
    theta_line_search_enabled=True,
    differentiable_mode=False,
)
res_theta = theta.solve(y0, f)

# RADAU
radau = RADAUSolver(
    t0=0.0, t1=10.0, dt=0.1,
    rtol=1e-6, atol=1e-8,
)
res_radau = radau.solve(y0, f)

# Compare final values
print(f"Theta final: {res_theta['final_state']}")
print(f"RADAU final: {res_radau['final_state']}")
print(f"Difference: {jnp.linalg.norm(res_theta['final_state'] - res_radau['final_state'])}")
```

---

## 5. Key Files

| Path | Purpose |
|------|---------|
| `NEOPAX/_transport_solvers.py` | Contains RADAUSolver, ThetaNewtonSolver, build_time_solver |
| `NEOPAX/tests/test_theta_solver_benchmarks.py` | Full benchmark test suite |
| `NEOPAX/tests/THETA_RADAU_BENCHMARKS.md` | Detailed documentation |

---

## 6. Common Use Cases

### Use Case: Stiff Transport Integration

```python
# Choose RADAU for offline/research: high accuracy, implicit, 5th order
solver = RADAUSolver(t0=0, t1=100, dt=1.0, rtol=1e-7, atol=1e-9, maxiter=100)
```

### Use Case: Production Real-Time Er Feedback

```python
# Choose Theta Robust: fast, handles branches, high control
params.theta_ptc_enabled = True           # Adaptive timestep
params.theta_line_search_enabled = True   # Robust acceptance
params.theta_homotopy_steps = 2           # Gradual forcing
```

### Use Case: Machine Learning / Sensitivity

```python
# Choose Theta Smooth: differentiable, smooth gradients
params.theta_differentiable_mode = True
params.theta_ptc_enabled = True
# Then pass through jax.grad for d(output)/d(parameters)
```

---

## 7. Performance Tips

1. **For high accuracy**: Use RADAU with `rtol=1e-8, atol=1e-10`
2. **For speed**: Use theta robust with `theta_ptc_enabled=True` and larger `dt`
3. **For stiffness**: RADAU handles $\lambda_{\max}/\lambda_{\min}$ ratios > 1000 easily
4. **For speed + stiffness**: Theta with `theta_gmres_tol=1e-6` reduces linear solves

---

## 8. Troubleshooting

**Q: RADAU solver diverges on my problem**  
A: Try increasing `maxiter=100` and loosening `tol=1e-6`

**Q: Theta smooth mode gives NaN gradients**  
A: Use theta robust mode → diffrax implicit adjoints for gradient

**Q: Benchmark tests fail to import NEOPAX modules**  
A: Ensure `PYTHONPATH` includes the NEOPAX root or run from `NEOPAX/` directory

**Q: "jax not installed" error**  
A: `pip install jax jaxlib jaxtyping`

---

## Next Steps

1. **Run the benchmark suite** to validate your setup
2. **Choose solver** based on your accuracy/speed tradeoff needs
3. **Read THETA_RADAU_BENCHMARKS.md** for detailed parameter tuning
4. **Add your own ODE** by extending `test_theta_solver_benchmarks.py`

---

