# RADAU Solver & Benchmark Tests Documentation

## Overview

This document describes the newly implemented RADAU5 implicit Runge-Kutta solver and the comprehensive benchmark test suite for evaluating theta-method and RADAU solvers on stiff ODE systems.

## Contents

1. [RADAU Solver Implementation](#radau-solver-implementation)
2. [Benchmark Equations](#benchmark-equations)
3. [Test Suite Structure](#test-suite-structure)
4. [Running the Tests](#running-the-tests)
5. [Results Interpretation](#results-interpretation)

---

## RADAU Solver Implementation

### Overview

**RADAU** is a 3-stage, 5th-order implicit Runge-Kutta method specifically designed for stiff ODEs. It provides:

- **5th-order accuracy**: $O(\Delta t^5)$ local truncation error
- **Excellent stability on stiff systems**: $L$-stable, handles eigenvalues with large negative real parts
- **Adaptive timestep control**: Embedded error estimation with safety factor
- **JAX-native implementation**: Fully jitable and differentiable via control flow

### Key Features

| Feature | Description |
|---------|-------------|
| **Implicit stages** | 3 Newton-iterable stages with Jacobian-Free inner iterations optional |
| **Order** | 5 (global), carefully designed for stiff transients |
| **Stability region** | Contains $(-\infty, 0)$ in complex plane; handles arbitrarily large negative eigenvalues |
| **Adaptive control** | Standard error estimation: $\text{err} \sim \|y_{\text{next}} - y_{\text{approx}}\|$ |
| **Differentiability** | Pure JAX control flow (lax.scan, lax.fori_loop) enables autodiff |
| **Jitability** | No Python conditionals; all branching via lax.cond |

### Butcher Tableau

```
c1 = (1 - √6)/10  ≈ 0.0885
c2 = (1 + √6)/10  ≈ 0.9114
c3 = 1.0

b = [5/120, -5/120, 100/120]  # Weights for final RK5 update
```

### Parameters

Constructor signature:
```python
RADAUSolver(
    t0=0.0,              # Start time
    t1=1.0,              # End time
    dt=0.1,              # Initial timestep
    rtol=1e-6,           # Relative tolerance for adaptive control
    atol=1e-8,           # Absolute tolerance for adaptive control
    max_step=1.0,        # Maximum timestep allowed
    min_step=1e-14,      # Minimum timestep before stopping
    tol=1e-8,            # Newton iteration convergence tolerance
    maxiter=50,          # Max Newton iterations per stage
    safety_factor=0.9,   # Multiplier for adaptive timestep
    min_step_factor=0.1, # Minimum contraction ratio per step
    max_step_factor=5.0, # Maximum growth ratio per step
)
```

### Integration

```python
from NEOPAX._transport_solvers import RADAUSolver

solver = RADAUSolver(t0=0, t1=10, dt=0.1, rtol=1e-6, atol=1e-8)

result = solver.solve(y0, vector_field)
# result['final_state']: y(t_final)
# result['ys']: all computed states (depending on save pattern)
# result['ts']: all time points
```

---

## Benchmark Equations

### 1. Scalar Stiff Decay

**Equation:**  
$$\frac{dy}{dt} = -\lambda y, \quad y(0) = y_0$$

**Exact Solution:**  
$$y(t) = y_0 e^{-\lambda t}$$

**Properties:**
- Simplest stiff test case ($\lambda = 100$)
- Tests basic implicit solver capabilities
- Linear system; no nonlinearities
- Used for convergence order verification

**Test file:** `ScalarStiffDecay` class

---

### 2. Forced Linear System

**Equation:**  
$$\frac{dy}{dt} = -\lambda y + \sin(\omega t), \quad y(0) = 0$$

**Exact Solution:** (via integrating factor)  
$$y(t) = \frac{\lambda \sin(\omega t) - \omega \cos(\omega t) + \omega e^{-\lambda t}}{\lambda^2 + \omega^2}$$

**Properties:**
- Moderately stiff ($\lambda = 50$)
- Non-homogeneous forcing; tests transient+steady-state response
- Analytical solution available for error norms
- Physical relevance: relaxation to periodic drive

**Test file:** `ForcedLinearSystem` class

---

### 3. Stiff 2×2 Linear System

**Equation:**  
$$\frac{d\mathbf{y}}{dt} = A \mathbf{y}, \quad A = \begin{pmatrix} -1 & 1 \\ 0 & -1000 \end{pmatrix}$$

**Exact Solution:** (matrix exponential)  
$$\mathbf{y}(t) = e^{At} \mathbf{y}_0$$

**Properties:**
- Highly stiff ($\lambda_1 = -1, \lambda_2 = -1000$; ratio = 1000)
- Tests 2D vector handling and eigvalue separation
- Dense Jacobian; classic benchmark from Hairer & Nørsett
- Analytical solution via matrix exponential

**Test file:** `StiffLinear2x2System` class

---

### 4. Logistic Equation

**Equation:**  
$$\frac{dy}{dt} = r y (1 - y/K), \quad y(0) = y_0$$

**Exact Solution:**  
$$y(t) = \frac{K}{1 + (K/y_0 - 1) e^{-rt}}$$

**Properties:**
- Nonlinear but non-stiff ($r = 1, K = 1$)
- Tests handling of smooth nonlinearities
- Bounded solution (saturation to $y = K$)
- Models population dynamics, autocatalytic reactions

**Test file:** `LogisticEquation` class

---

### 5. Stiff Branchy Toy Model (3D)

**Equations:**  
$$\frac{dy_1}{dt} = -100 y_1 + 10 \tanh(y_3)$$
$$\frac{dy_2}{dt} = -0.1(y_2 - y_1)$$
$$\frac{dy_3}{dt} = 10(y_1 - 2y_3 + y_1^3)$$

**Properties:**
- **No analytical solution** (nonlinear coupling)
- Stress test: three distinct timescales (100, 0.1, 10)
- Branchy transient via cubic nonlinearity
- Tanh coupling creates sharp transitions
- Designed for stellarator $E_r$ relaxation dynamics
- Used for stability, non-explosion, and steady-state validation

**Validation approach:**
1. Numerical steady state via large-time integration
2. Residual $\|\mathbf{f}(t, \mathbf{y}_\infty)\|$ should be $< 10^{-6}$
3. Solution bounded: $\|\mathbf{y}(t)\| < 10$ for all $t$

**Test file:** `StiffBranchyToyModel` class

---

## Test Suite Structure

### Location
```
NEOPAX/tests/test_theta_solver_benchmarks.py
```

### Test Classes

| Class | Fixture | Equations | Purpose |
|-------|---------|-----------|---------|
| `TestScalarDecay` | `scalar_decay_problem` | #1 | Baseline stiff solver validation |
| `TestForcedLinear` | `forced_linear_problem` | #2 | Transient + periodic response |
| `TestStiff2x2Linear` | `stiff_linear_2x2_problem` | #3 | Extreme stiffness, vector handling |
| `TestLogisticEquation` | `logistic_problem` | #4 | Mild nonlinearity |
| `TestBranchyStiffToyModel` | `branchy_problem` | #5 | Stress test, branchy transients |
| `TestConvergence` | (mixed) | #1, #2, #3 | Step-refinement convergence orders |
| `TestDifferentiability` | (mixed) | #1 | jit/grad through solvers |

### Fixtures

**Solver Fixtures:**
```python
@pytest.fixture
def theta_solver_robust():
    """Robust theta-method with PTC, retries, hard accept/reject."""
    ...

@pytest.fixture
def theta_solver_smooth():
    """Smooth theta-method in differentiable mode."""
    ...

@pytest.fixture
def radau_solver():
    """RADAU method with adaptive timestep."""
    ...
```

**Problem Fixtures:**
```python
@pytest.fixture
def scalar_decay_problem():
    """Scalar stiff decay benchmark."""
    return ScalarStiffDecay(lam=100.0)

@pytest.fixture
def branchy_problem():
    """3D stiff branchy stress test."""
    return StiffBranchyToyModel()
```

---

## Running the Tests

### Prerequisites
```bash
pip install pytest jax jaxtyping diffrax
```

### Full Test Suite
```bash
cd NEOPAX
pytest tests/test_theta_solver_benchmarks.py -v
```

### Run Specific Test Class
```bash
pytest tests/test_theta_solver_benchmarks.py::TestScalarDecay -v
```

### Run Single Test
```bash
pytest tests/test_theta_solver_benchmarks.py::TestScalarDecay::test_theta_robust_scalar_decay -v
```

### With Output & Debugging
```bash
pytest tests/test_theta_solver_benchmarks.py -v -s --tb=short
```

### Mark-Based Filtering
```bash
# Run only convergence tests
pytest tests/test_theta_solver_benchmarks.py -k "convergence" -v

# Skip differentiability tests (if JAX grad issues expected)
pytest tests/test_theta_solver_benchmarks.py -k "not differentiability" -v
```

---

## Results Interpretation

### Expected Behavior

| Equation | Theta (Robust) | RADAU | Notes |
|----------|---|---|---|
| Scalar decay | Error < $10^{-3}$ | Error < $10^{-5}$ | RADAU much more accurate for stiff |
| Forced linear | Error < $0.01$ | Error < $10^{-4}$ | RADAU benefits from 5th order |
| 2×2 stiff | Rel.error < 1% | Rel.error < $10^{-4}$ | High stiffness ratio faors RADAU |
| Logistic | Error < $0.01$ | Error < $10^{-5}$ | Both should handle smoothly |
| Branchy 3D | Bounded, residual < $10^{-6}$ | Bounded, residual < $10^{-8}$ | RADAU more accurate transient |

### Convergence Order

For step refinement $\Delta t \to \Delta t/2$, error ratio should be:
- **Backward Euler** (theta method): Ratio ≈ 2 (1st order)
- **Theta generic**: Ratio = 2–4 (1st–2nd order)
- **RADAU5**: Ratio ≈ 32 (5th order, if grid sufficiently fine)

**Test validates**: `test_theta_step_refinement_scalar_decay`

### Gradient Behavior

**Tests check:**
1. `test_grad_through_scalar_decay_theta_smooth`: Gradient through autodiff path
2. `test_jit_through_solver`: JIT compilation of entire solver

**Expected:**
- Gradients should be finite and non-NaN
- Gradient sign should make physical sense (e.g., negative for decay acceleration)
- JIT should not error; output should match non-jitted version

---

## Performance Characteristics

### Theta-Method (Robust Mode)
- **Pros**: Robust, handles sharp transitions, large steps possible
- **Cons**: 1st–2nd order, requires more steps for accuracy
- **Best for**: Stellarator Er problems with discontinuities

### Theta-Method (Smooth/Differentiable Mode)
- **Pros**: Smooth gradients, differentiable, jitable
- **Cons**: May require smaller steps, soft logic instead of hard acceptance
- **Best for**: Machine learning, sensitivity analysis, neural ODE training

### RADAU Method
- **Pros**: 5th-order accurate, excellent stiff stability, fewer steps needed
- **Cons**: Implicit stages (3 Newton solves per step), more function evaluations
- **Best for**: High-accuracy requirement, very stiff transients, offline simulations

---

## Extension Points

### Adding New Benchmark Equations

```python
class MyEquation:
    """Template for new benchmark."""
    
    def __call__(self, t, y):
        """Compute dy/dt."""
        return ...
    
    def exact(self, t, y0):
        """Return y(t) if analytical solution known."""
        return ...
    
    def error_at_t(self, t, y_final, y0):
        """Return |y_final - exact(t, y0)|."""
        return jnp.abs(y_final - self.exact(t, y0))
```

Then add fixture and test class:
```python
@pytest.fixture
def my_problem():
    return MyEquation(...)

class TestMyProblem:
    def test_theta_robust_my_problem(self, my_problem):
        ...
```

### Customizing Solver Parameters

```python
# Override default theta parameters
params = Solver_Parameters()
params.theta_ptc_growth = 2.0        # More aggressive growth
params.theta_gmres_maxiter = 300     # Better linear solver control
params.theta_homotopy_steps = 3      # More homotopy continuation stages

solver = build_time_solver(params, 'theta')
```

---

## References

- Hairer & Nørsett, *Solving Ordinary Differential Equations I*, 2nd ed. (Springer, 1993)
- Butcher, *Numerical Methods for Ordinary Differential Equations* (Wiley, 2nd ed. 2008)
- Giladi & Keller, "Asymptotically stable Gauss-Radau quadrature rules" (examples of RADAU geometry)

---

## Common Issues & Troubleshooting

### Issue: Tests fail with "JAX not installed"
**Solution:** `pip install jax jaxlib jaxtyping`

### Issue: RADAU solver diverges on branchy problem
**Cause:** Newton iteration tolerance might be too strict or max iter too low
**Solution:** Increase `maxiter=100`, loosen `tol=1e-6`

### Issue: Theta smooth mode produces NaN in gradients
**Cause:** Line-search sigmoid might be saturated or Jacobian singular
**Solution:** Switch to robust mode (differentiable_mode=False) for forward solve, then

 use implicit adjoints

### Issue: Convergence order test fails to show 5th order for RADAU
**Cause:** Grid might be too coarse; higher-order methods reveal error at finer scales
**Solution:** Use smaller dt and test over very small error regime (atol < 1e-10)

---

