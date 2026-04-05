# Architecture & Integration Guide

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    NEOPAX Transport Solver System                    │
└─────────────────────────────────────────────────────────────────────┘

User Config (TOML or Parameters)
        │
        ▼
┌──────────────────────────┐
│  Solver_Parameters       │
│  ├─ transport_solver_    │
│  │  backend: "radau"     │
│  ├─ t0, t_final, dt      │
│  ├─ rtol, atol           │
│  └─ ...                  │
└──────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│  build_time_solver()                    │
│  ├─ Parses backend string ───┐         │
│  ├─ Creates solver instance  │         │
│  └─ Returns TransportSolver ◄────┐     │
└─────────────────────────────────┼──────┘
                                  │
                        ┌─────────▼────────┐
                        │ ODE_SOLVER_      │
                        │ BACKENDS set:    │
                        │ ├─ "radau" ◄─────┼─── NEW!
                        │ ├─ "diffrax_*"   │
                        │ ├─ "predictor_"  │
                        │ │   "corrector"  │
                        │ └─ "heun"        │
                        └──────────────────┘
                        ┌─────────▼────────┐
                        │ TIME_SOLVER_     │
                        │ REGISTRY:        │
                        │ ├─ "radau" ──┐   │
                        │ │   ↓ func   │   │
                        │ │ RADAUSolver│
                        │ ├─ "theta_"  │   │
                        │ │ "newton"   │   │
                        │ └─ ...       │   │
                        └──────────────────┘
        │
        ▼
    ┌─────────────────────────┐
    │  Solver Instance        │
    │  (TransportSolver)      │
    │  ┌─────────────────┐    │
    │  │ RADAUSolver or  │    │
    │  │ ThetaNewtonSolver│   │
    │  │ or other        │    │
    │  └─────────────────┘    │
    └──────────┬──────────────┘
               │
               ▼
    ┌────────────────────────────────────┐
    │  solver.solve(y0, f, *args)        │
    │  ├─ Integrate ODE: y' = f(t, y)    │
    │  ├─ Support JAX operations:        │
    │  │  ├─ jax.jit() (full)            │
    │  │  ├─ jax.grad() (smooth mode)    │
    │  │  ├─ jax.vmap() (batch)          │
    │  │  └─ jax.jacfwd() (Jacobian)     │
    │  └─ Return result dict             │
    │     ├─ final_state                 │
    │     ├─ ys (all steps)              │
    │     ├─ ts (time points)            │
    │     └─ metadata                    │
    └────────────────────────────────────┘
               │
               ▼
        ┌──────────────┐
        │  Integrated  │
        │    Solution  │
        └──────────────┘
```

---

## RADAU Implementation Details

```
┌─────────────────────────────────────────────────────────────┐
│                    RADAUSolver.solve()                       │
│  (Full JAX control flow: lax.scan, lax.fori_loop)           │
└──────────────────┬──────────────────────────────────────────┘
                   │
        ┌──────────▼──────────┐
        │ Main time loop      │
        │ (lax.scan)          │
        │ ├─ Initialize: t0, y0, dt0
        │ ├─ Loop until t >= t_final
        │ │
        │ └─► _radau_step()   │◄───────┐
        │                     │        │
        └─────────────────────┘        │
                                       │
                    ┌──────────────────┘
                    │
        ┌───────────▼──────────────────┐
        │ _radau_step()                 │
        │ ├─ Stage 1: Compute k1       │
        │ ├─ Stage 2: Compute k2       │
        │ ├─ Stage 3: Compute k3       │
        │ ├─ Update: y_next ← weighted │
        │ │           sum of k_i       │
        │ └─ Error estimate & adapt dt │
        │                              │
        └──┬──────────────────────────┘
           │
        ┌──▼──────────────────────────────┐
        │ Per-stage implicit solve         │
        │ (lax.fori_loop Newton)           │
        │ ├─ Residual: R(k) = k - f(...)  │
        │ ├─ Newton: δk = -J^{-1} R       │
        │ ├─ Update: k ← k + δk           │
        │ └─ Iterate until converged      │
        └──┬──────────────────────────────┘
           │
        ┌──▼─────────────────────────────┐
        │ Jacobian computation            │
        │ (direct solve)                  │
        │ jac = jax.jacfwd(residual)      │
        │ δk = jax.linalg.solve(jac, -R) │
        └─────────────────────────────────┘
```

---

## Test Suite Architecture

```
┌───────────────────────────────────────────────────────────┐
│           test_theta_solver_benchmarks.py                  │
└──────────────────┬────────────────────────────────────────┘
                   │
        ┌──────────┴────────────────────────┐
        │                                   │
        ▼                                   ▼
   PROBLEM CLASSES               SOLVER INSTANCES
   
   ScalarStiffDecay              theta_solver_robust
   ├─ __call__(t, y)             theta_solver_smooth
   ├─ exact(t, y0)               radau_solver
   └─ error_at_t(...)
   
   ForcedLinearSystem
   StiffLinear2x2System
   LogisticEquation
   StiffBranchyToyModel
   
        ▼                                   ▼
   
   ┌─────────────────────────────────────────────┐
   │     @pytest.fixture decorators              │
   │     (Parameterized problem & solver setup)  │
   └─────────────────────────────────────────────┘
        │
        ▼
   ┌──────────────────────────────────────────────┐
   │           TEST CLASSES                       │
   ├──────────────────────────────────────────────┤
   │ TestScalarDecay                              │
   │ ├─ test_theta_robust_scalar_decay()          │
   │ ├─ test_theta_smooth_scalar_decay()          │
   │ └─ test_radau_scalar_decay()                 │
   │                                              │
   │ TestForcedLinear (2 methods)                 │
   │ TestStiff2x2Linear (2 methods)               │
   │ TestLogisticEquation (2 methods)             │
   │ TestBranchyStiffToyModel (3 methods)         │
   │ TestConvergence (1 method)                   │
   │ TestDifferentiability (2 methods)            │
   └──────────────────────────────────────────────┘
        │
        ▼
   ┌──────────────────────────────────────────────┐
   │        pytest Test Execution                 │
   │ ├─ Discover test_*.py files                  │
   │ ├─ Match Test* classes & test_* methods      │
   │ ├─ Inject fixtures                           │
   │ ├─ Run assertions                            │
   │ └─ Report pass/fail                          │
   └──────────────────────────────────────────────┘
```

---

## Integration Flow: From Config to Solve

### Scenario: User selects RADAU via TOML config

```yaml
[solver]
transport_solver_backend = "radau"
t0 = 0.0
t_final = 100.0
dt = 0.1
rtol = 1e-6
atol = 1e-8
```

**Execution Flow:**

```
1. TOML parsed → Python dict
   
2. Create Solver_Parameters:
   params = Solver_Parameters(
       transport_solver_backend="radau",
       t0=0.0, t_final=100.0, dt=0.1,
       rtol=1e-6, atol=1e-8, ...
   )

3. Call build_time_solver(params):
   a) family, backend = _select_solver_family_and_backend(params)
      └─► family="ode", backend="radau"
   
   b) Check backend in ODE_SOLVER_BACKENDS
      └─► "radau" found ✓
   
   c) Look up registry: TIME_SOLVER_REGISTRY["radau"]
      └─► Returns λ **kw: RADAUSolver(**kw)
   
   d) Extract parameters:
      t0, t1, dt = 0.0, 100.0, 0.1
      rtol, atol = 1e-6, 1e-8
   
   e) Return RADAUSolver(
          t0=0.0, t1=100.0, dt=0.1,
          rtol=1e-6, atol=1e-8, ...
      )

4. solver = build_time_solver(params)
   └─► solver is RADAUSolver instance

5. result = solver.solve(y0, vector_field)
   └─► Full JAX-native integration with adaptive timestep

6. Access solution:
   y_final = result['final_state']
   y_all = result['ys']
   t_all = result['ts']
```

---

## Extensibility: Adding a New ODE Solver

### Template

```python
# Step 1: Define solver class in _transport_solvers.py
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class MyCustomSolver(TransportSolver):
    t0: float
    t1: float
    dt: float
    # ... custom parameters
    
    def __init__(self, t0, t1, dt, **kwargs):
        # Custom init logic
        pass
    
    def solve(self, state, vector_field: Callable, *args, **kwargs):
        # Implementation using lax.scan, lax.fori_loop, lax.while_loop
        # Return dict: {'final_state': ..., 'ys': ..., ...}
        pass

# Step 2: Register in ODE_SOLVER_BACKENDS
ODE_SOLVER_BACKENDS.add("my_solver")

# Step 3: Register factory
register_time_solver(
    "my_solver",
    lambda **kw: MyCustomSolver(**kw)
)

# Step 4: Now available via config
# transport_solver_backend = "my_solver"

# Step 5: Add tests
# tests/test_theta_solver_benchmarks.py
class TestMyCustomSolver:
    def test_scalar_decay(self, scalar_decay_problem):
        solver = MyCustomSolver(t0=0, t1=1, dt=0.1)
        result = solver.solve(jnp.array([1.0]), scalar_decay_problem)
        assert jnp.abs(result['final_state'][0] - expected) < tol
```

---

## JAX Compatibility Matrix

### Features Supported by Implementation Type

| Feature | RADAU | Theta (Robust) | Theta (Smooth) |
|---------|-------|---|---|
| Forward solve | ✓ | ✓ | ✓ |
| jax.jit | ✓ | ✓ | ✓ |
| jax.grad | ✓ (limited) | ✗ | ✓ |
| jax.vmap | ~ | ~ | ✓ |
| jax.jacfwd | ✓ (limited) | ✗ | ✓ |
| Adaptive dt | ✓ | ✓ (PTC) | ✗ |
| Production robust | ✓ | ✓ | ~ |
| Sensitivity analysis | ~ | ✗ | ✓ |
| Neural ODE | ~ | ✗ | ✓ |

✓ = Fully supported | ~ = Partially supported | ✗ = Not supported

---

## Documentation Cross-Reference

| Document | Audience | Content |
|----------|----------|---------|
| **IMPLEMENTATION_SUMMARY.md** | Developers | What was built, design decisions, file structure |
| **THETA_RADAU_BENCHMARKS.md** | Researchers | Detailed RADAU math, benchmark equations, tuning |
| **RADAU_QUICK_START.md** | Users | Copy-paste examples, test commands, troubleshooting |
| **ARCHITECTURE.md** (this file) | Integration team | System diagrams, extensibility, JAX compatibility |

---

## Compatibility with Existing NEOPAX

### Backward Compatibility

✅ **Fully backward compatible:**
- Existing theta solvers unchanged
- Existing nonlinear solvers unchanged
- No changes to _parameters.py (RADAU uses standard kwargs)
- No changes to run_from_toml.py (reads RADAU backend naturally)

### Migration Path

**Before (using diffrax Kvaerno5):**
```python
params.transport_solver_backend = "diffrax_kvaerno5"
```

**After (optionally switching to RADAU):**
```python
params.transport_solver_backend = "radau"
```

**Or keeping old solver:**
```python
params.transport_solver_backend = "diffrax_kvaerno5"  # Still works!
```

---

## Performance Tuning Guide

### RADAU Parameter Tuning

```python
# Conservative (high accuracy, slower)
RADAUSolver(
    rtol=1e-8, atol=1e-10,
    maxiter=100,
    max_step_factor=2.0,
)

# Balanced (default)
RADAUSolver(
    rtol=1e-6, atol=1e-8,
    maxiter=50,
    max_step_factor=5.0,
)

# Aggressive (speed, adequate accuracy)
RADAUSolver(
    rtol=1e-4, atol=1e-6,
    maxiter=30,
    max_step_factor=10.0,
)
```

### Theta Parameter Tuning

```python
# Highly stiff (e.g., 1000:1 eigenvalue ratio)
ThetaNewtonSolver(
    theta_ptc_growth=1.2,         # Conservative growth
    theta_ptc_shrink=0.5,         # Aggressive shrink
    theta_homotopy_steps=3,       # Gradual forcing
    theta_gmres_tol=1e-6,         # Looser linear solver
    theta_max_step_retries=15,    # More retries
)

# Moderate stiffness (default)
ThetaNewtonSolver(
    theta_ptc_growth=1.5,
    theta_ptc_shrink=0.5,
    theta_homotopy_steps=1,
    theta_gmres_tol=1e-8,
    theta_max_step_retries=8,
)

# Production (fast path, lower accuracy tolerance)
ThetaNewtonSolver(
    theta_ptc_growth=2.0,
    theta_ptc_shrink=0.75,
    theta_homotopy_steps=0,       # Skip homotopy
    theta_gmres_tol=1e-5,
    theta_max_step_retries=5,
    differentiable_mode=False,    # Faster (no soft damping)
)
```

---

## Verification Checklist

- [x] RADAU solver implements 3-stage RADAU5 correctly
- [x] All 5 analytical benchmark equations with exact solutions
- [x] Branchy 3D toy model for stress testing
- [x] 14+ comprehensive test methods
- [x] Tests pass (or degrade gracefully in missing-dependency environments)
- [x] RADAUSolver fully jitable (no Python conditionals in main loop)
- [x] RADAU solver differentiable (no hard accept/reject branching)
- [x] Dual-level documentation (technical + quick-start)
- [x] Backward compatible with existing solvers
- [x] Registry updated with new backend

---

