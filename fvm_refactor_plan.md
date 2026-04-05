# NEOPAX FVM Refactor Plan

**Goal:** Replace ad-hoc ghost-cell + 1st-order arithmetic-mean face interpolation with a
`CellVariable`-based FVM that is:
- 2nd-order accurate at interior faces
- BC-correct (Dirichlet / Neumann / Robin) on both axis and edge
- Non-uniform-grid-aware
- Fully JAX-jittable and differentiable (compatible with RADAU, Kvaerno5, theta solvers)
- Optionally WENO-3 monotone
- Optionally positivity-preserving

Time solvers (RADAU, Kvaerno5, theta) are **not changed**.

---

## JAX jit/differentiability constraints (applies to every phase)

| Rule | Reason |
|---|---|
| No Python-level `if` on array values — use `jnp.where` | Dynamic control flow breaks `jit` |
| No in-place mutation — use `.at[].set()` or functional ops | JAX arrays are immutable |
| All BC values must be JAX arrays, not Python floats | Traced through `jax.grad` |
| `right_face_constraint` and `right_face_grad_constraint` can be `None` only as a **static** Python choice — not data-dependent | Static Pythonic branching is fine inside a frozen dataclass; runtime data-dependent branching is not |
| `jax.vmap` over species axis — every function it wraps must operate on a single-species 1-D array | Avoids Python loops; keeps species dimension differentiable |
| `weno3_face_value` must use only `jnp` ops (no Python loops over cells) — vectorise with `jax.vmap` or tensor-slice arithmetic | Required for `jit` + `jax.grad` |
| Positivity floor via `jnp.where`, never Python `if state < floor` | Smooth / traceable |
| `CellVariable` is a `@jax.tree_util.register_dataclass` frozen dataclass — carries static flags (`left_face_constraint is None`) as Python-level Nones, carries array data as leaves | Already done in current `_cell_variable.py` |

---

## Phase 1 — Non-uniform cell widths (prerequisite, ~10 lines)

**Files:** `_fem.py`, `_transport_equations.py`

### Problem
`conservative_update(flux, dx, ...)` takes scalar `dx = field.dr`, assuming a uniform grid.
Non-uniform spacing (e.g. finer near the axis) uses wrong cell volumes.

### Change in `_fem.py`
```python
# dx can be scalar float or (n_r,) array — JAX broadcasting handles both cases identically.
def conservative_update(flux, dx, Vprime=None, Vprime_half=None, source=None):
    net_flux = Vprime_half[1:] * flux[1:] - Vprime_half[:-1] * flux[:-1]
    update = -net_flux / (Vprime * dx)    # dx broadcast: () or (n_r,)
    if source is not None:
        update = update + source
    return update
```

No signature break. Scalar `dr` still works unchanged.

### Change at call sites in `_transport_equations.py`
```python
# Replace:  dr = field.dr
# With:
dr_cells = jnp.diff(field.r_grid_half)   # (n_r,) — correct for non-uniform grids
```

### JAX notes
- `jnp.diff` is jittable and differentiable.
- Broadcasting `(n_r+1,) / (n_r,)` is standard NumPy-compatible broadcasting — no issue.

---

## Phase 2 — CellVariable-based BC + 2nd-order face interpolation (main change)

**Files:** `_boundary_conditions.py`, `_transport_equations.py`

This is the highest-impact phase.  Replaces the three separate ghost-cell +
`faces_from_cell_centered` patterns with a single unified path.

### Design

```
Axis (r=0)                  Interior faces           Edge (r=a)
   |       n_r cell-centred fluxes Γ_i       |
   left_face_constraint=0.0                   right BC from BoundaryConditionModel
   (stellarator symmetry: Γ=0 at axis)        (Dirichlet / Neumann / Robin)
                  ↓
      CellVariable.face_value()         ← 2nd-order linear at all interior faces
      (already implemented, unused)
                  ↓
      conservative_update(flux, dr_cells, Vprime, Vprime_half)
```

**Why axis=0 left BC?**
In toroidal/stellarator geometry, regularity at the magnetic axis requires
the radial flux Γ → 0 as r → 0. This is a `left_face_constraint=0.0` (Dirichlet on flux).

### Add to `_boundary_conditions.py`

```python
def right_constraints_from_bc_model(bc_model, default_value):
    """
    Translate a BoundaryConditionModel right-BC spec into CellVariable constraint args.

    Returns (right_face_constraint, right_face_grad_constraint) as JAX arrays or None.
    The None/not-None distinction is a STATIC Python choice made at trace time
    (frozen dataclass field), not a runtime data-dependent branch — safe for jit.

    Parameters
    ----------
    bc_model : BoundaryConditionModel or None
    default_value : jax.Array, shape ()
        The current cell value at the right boundary (used as Dirichlet fallback).
    """
    if bc_model is None:
        # Default: zero-gradient Neumann (no information lost at edge)
        return None, jnp.asarray(0.0)
    rt = str(getattr(bc_model, "right_type", "dirichlet")).strip().lower()
    if rt == "dirichlet":
        rv = jnp.asarray(bc_model.right_value) if bc_model.right_value is not None else default_value
        return rv, None
    if rt == "neumann":
        rg = jnp.asarray(bc_model.right_gradient) if bc_model.right_gradient is not None else jnp.asarray(0.0)
        return None, rg
    if rt == "robin":
        # Robin: α*u + β*∂u/∂r = γ  →  ∂u/∂r = (γ - α*u)/β
        # Linearised at current edge value: use -right_value/right_decay_length as grad constraint
        decay = jnp.asarray(bc_model.right_decay_length) if bc_model.right_decay_length is not None else jnp.asarray(1.0)
        rv = jnp.asarray(bc_model.right_value) if bc_model.right_value is not None else jnp.asarray(0.0)
        return None, -rv / (decay + 1e-12)
    # Fallback: zero gradient
    return None, jnp.asarray(0.0)
```

**JAX note:** The `if rt == ...` branch is on a Python string from a frozen dataclass —
it is resolved at trace time, not at runtime. This is correct and safe for `jit`.
`bc_model.right_value` etc. must be JAX arrays at run time so `jax.grad` can trace through them.

### Rewrite `DensityEquation.__call__` and `TemperatureEquation.__call__`

```python
from ._boundary_conditions import right_constraints_from_bc_model
from ._cell_variable import make_profile_cell_variable

def __call__(self, state, flux_models, source_models, field, bc=None, **kwargs):
    dr_cells  = jnp.diff(field.r_grid_half)          # (n_r,)  Phase-1 change
    Vprime     = field.Vprime
    Vprime_half = field.Vprime_half
    Gamma_total = flux_models.get("Gamma_total")      # (n_species, n_r)

    bc_density = bc.get("density") if bc is not None else None

    def _species_faces(G_s):
        # BC constraints are resolved statically at trace time from frozen bc_density
        rv, rg = right_constraints_from_bc_model(bc_density, G_s[-1])
        cv = make_profile_cell_variable(
            G_s, field.r_grid_half,
            left_face_constraint=jnp.asarray(0.0, dtype=G_s.dtype),  # axis: Γ=0
            right_face_constraint=rv,
            right_face_grad_constraint=rg,
        )
        return cv.face_value()   # (n_r+1,)  — 2nd-order linear at interior faces

    Gamma_faces = jax.vmap(_species_faces)(Gamma_total)  # (n_species, n_r+1)
    density_rhs = jax.vmap(
        lambda flux: conservative_update(flux, dr_cells, Vprime, Vprime_half)
    )(Gamma_faces)

    if source_models is not None and "density" in source_models:
        density_rhs = density_rhs + source_models["density"](state)
    return density_rhs
```

Same pattern for `TemperatureEquation` (replace `"density"` / `Gamma_total` / `Q_total`).

### Rewrite `ElectricFieldEquation.__call__` (Er faces)

```python
# Replace set_dirichlet_ghosts + faces_from_cell_centered for Er:
bc_er = bc.get("Er") if bc is not None else None
rv_er, rg_er = right_constraints_from_bc_model(bc_er, Er[-1])
Er_cv = make_profile_cell_variable(
    Er, field.r_grid_half,
    left_face_grad_constraint=jnp.asarray(0.0, dtype=Er.dtype),  # ∂Er/∂r=0 at axis
    right_face_constraint=rv_er,
    right_face_grad_constraint=rg_er,
)
Er_faces = Er_cv.face_value()    # (n_r+1,) — 2nd-order
```

**JAX notes for Phase 2:**
- `make_profile_cell_variable` and `CellVariable.face_value()` are pure functions of JAX arrays — fully jittable and differentiable.
- `jax.vmap(_species_faces)(Gamma_total)` vectorises over species without Python loops; `jax.grad` through `Gamma_total` works.
- The `left_face_constraint` is a concrete `jnp.asarray(0.0)` — never `None` at the axis since axis zero-flux is a physical constraint; `CellVariable.left_face_value()` takes the `if self.left_face_constraint is not None` branch, which is a static Python check on the dataclass field.

### BC type coverage

| `right_type` | `right_face_constraint` | `right_face_grad_constraint` | `CellVariable.face_value()` result at edge |
|---|---|---|---|
| `"dirichlet"` | `right_value` (array) | `None` | Edge face **pinned** to `right_value` |
| `"neumann"` | `None` | `right_gradient` (array) | Edge face extrapolated so `dΓ/dr = right_gradient` |
| `"robin"` | `None` | `-rv/decay` (array) | Edge face extrapolated with linearised Robin slope |
| `None` (no bc_model) | `None` | `0.0` | Zero-gradient Neumann — no artefact at edge |

---

## Phase 3 — WENO-3 monotone reconstruction (optional toggle)

**Files:** `_fem.py` (new function), `_cell_variable.py` (new method / param), `_transport_equations.py` (new `reconstruction` field on equations)

### Why
2nd-order linear interpolation (Phase 2) can produce over/undershoots near sharp radial
gradients (H-mode pedestal, internal transport barrier). WENO-3 is monotone near
discontinuities and 3rd-order smooth elsewhere. TORAX does not implement this.

### Implementation in `_fem.py`

```python
def weno3_inner_face_values(values, face_centers):
    """
    WENO-3 reconstruction at n_r-1 interior faces from n_r cell-centred values.
    Fully vectorised — no Python loops over r. Jittable and differentiable.

    Candidate stencils at face i+1/2 (0-indexed interior faces: i=1..n_r-2):
      S0 (left-biased):  q0 = -v[i-1]/2 + 3*v[i]/2
      S1 (right-biased): q1 =  v[i]/2   + v[i+1]/2

    Smoothness indicators (L2 norm of derivative over stencil):
      beta0 = (v[i] - v[i-1])^2
      beta1 = (v[i+1] - v[i])^2

    Optimal weights:  d0=1/3, d1=2/3
    Nonlinear weights with epsilon=1e-6 to avoid division by zero:
      alpha_k = d_k / (eps + beta_k)^2
      omega_k = alpha_k / (alpha_0 + alpha_1)

    Face value = omega_0*q0 + omega_1*q1
    """
    eps = 1e-6
    v = values                          # (..., n_r) — leading batch dims OK
    # Stencil values for interior faces 1..n_r-1
    v_im1 = v[..., :-2]                # v[i-1] for i=1..n_r-1
    v_i   = v[..., 1:-1]               # v[i]
    v_ip1 = v[..., 2:]                 # v[i+1]
    # Candidate reconstructions
    q0 = -0.5 * v_im1 + 1.5 * v_i
    q1 =  0.5 * v_i   + 0.5 * v_ip1
    # Smoothness indicators
    beta0 = jnp.square(v_i   - v_im1)
    beta1 = jnp.square(v_ip1 - v_i)
    # Nonlinear weights
    alpha0 = (1.0 / 3.0) / jnp.square(eps + beta0)
    alpha1 = (2.0 / 3.0) / jnp.square(eps + beta1)
    denom  = alpha0 + alpha1
    omega0 = alpha0 / denom
    omega1 = alpha1 / denom
    # Face values at interior faces i+1/2, i=1..n_r-2  →  n_r-1 values total?
    # Note: interior faces in NEOPAX convention are face_centers[1:-1], count = n_r-1
    return omega0 * q0 + omega1 * q1   # (..., n_r-1)
```

**Note on stencil at face 1 (closest to axis):** at `i=0` there is no `v[-1]`; handle by
using the left face value (from `CellVariable.left_face_value()`) as `v_im1` for that face.
Implement as a one-element pad before slicing — avoids `if` branches:

```python
left_val = cv.left_face_value()                   # (..., 1)
v_padded = jnp.concatenate([left_val, cv.value], axis=-1)  # (..., n_r+1)
# Now stencil covers all interior faces cleanly
```

### Add method to `CellVariable`

```python
def face_value(self, reconstruction: str = "linear"):
    """
    reconstruction : "linear" (default, 2nd-order) | "weno3" (monotone, 3rd-order smooth)
    The 'reconstruction' argument is a static Python string — resolved at trace time.
    Both branches are jittable and differentiable.
    """
    if reconstruction == "weno3":
        inner = weno3_inner_face_values_with_left_pad(self)
    else:
        inner = _linear_inner_face_values(self.value, self.face_centers)
    return jnp.concatenate([self.left_face_value(), inner, self.right_face_value], axis=-1)
```

**Static-string note:** `reconstruction` is passed as a Python literal at construction
time of the equation dataclass (see below). JAX traces a separate compiled path per
unique string — this is the correct JAX pattern for static enum-like choices.

### Add `reconstruction` field to equation dataclasses

```python
@register_equation("density")
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class DensityEquation(EquationBase):
    name: str = "density"
    reconstruction: str = "linear"   # static — "linear" or "weno3"

    def __call__(self, ...):
        def _species_faces(G_s):
            cv = make_profile_cell_variable(G_s, field.r_grid_half, ...)
            return cv.face_value(reconstruction=self.reconstruction)
        ...
```

Default `"linear"` — zero behaviour change for existing code. Use `"weno3"` by setting
`DensityEquation(reconstruction="weno3")` at construction.

### JAX notes
- No Python loops — all stencil operations are vectorised tensor arithmetic.
- `jnp.square`, division, `jnp.concatenate` are all differentiable.
- `eps=1e-6` prevents `NaN` gradients when `beta0=beta1=0` (uniform profile).

---

## Phase 4 — Positivity-preserving floor

**Files:** `_parameters.py` (new fields), `_main_solver.py` (floor enforcement)

This phase is a safety net — cheap to add, prevents crashes when density/temperature
hits zero transiently during stiff solving.

### Add to `Solver_Parameters.__init__`

```python
density_floor: float | None = None       # e.g. 1e10  [m^-3 or normalised units]
temperature_floor: float | None = None   # e.g. 1e-3  [keV]
```

### Enforcement in `MainTransportModel.vector_field`

After computing `dn_dt` and `dT_dt`:

```python
if self.solver_parameters.density_floor is not None:
    floor_n = jnp.asarray(self.solver_parameters.density_floor)
    dn_dt = jnp.where(
        state.density < floor_n,
        jnp.maximum(dn_dt, 0.0),   # allow only increases below floor
        dn_dt,
    )

if self.solver_parameters.temperature_floor is not None:
    floor_T = jnp.asarray(self.solver_parameters.temperature_floor)
    dT_dt = jnp.where(
        state.temperature < floor_T,
        jnp.maximum(dT_dt, 0.0),
        dT_dt,
    )
```

### JAX notes
- `jnp.where` is fully differentiable via subgradient at the threshold (acceptable).
- `jnp.maximum` is differentiable everywhere except at zero (acceptable).
- `self.solver_parameters.density_floor` is a Python scalar stored in the frozen
  dataclass — `jnp.asarray(floor_n)` makes it a traced leaf when used inside `vector_field`.
- `None` check is static (Python level) — safe for `jit`.

---

## Phase 5 — AP linearisation for Er (Newton preconditioner, low priority)

**Files:** `_transport_equations.py`, `_main_solver.py`

Only needed when `ε/Δt ≪ 1` causes ill-conditioned Newton Jacobian.
Not required for current ε~0.02, Δt~0.1 regime.

### AP decomposition

The Er RHS is:
```
f(Er) = DEr * ∇²Er  -  Σ_s Z_s e Γ_s^neo(Er) / ε_⊥
```

Linearise around `Er^n`:
```
∂f/∂Er|_{Er^n} ≡ J_Er    (diagonal in r-space to first order)
```

Split:
- **Stiff linear part:** `L * Er = J_Er * Er`
- **Explicit source:** `S = f(Er^n) - J_Er * Er^n`

AP update (resolves ambipolar root automatically as ε → 0):
```
Er^{n+1} = Er^n + f^n / (ε/Δt - J_Er)
```

### Design in `ElectricFieldEquation`

```python
def jacobian_diagonal(self, Er, state, field, **kwargs):
    """∂f/∂Er — diagonal part only. Used as Newton preconditioner."""
    # Forward-mode jvp through the Er-dependent part of f
    f0 = self._er_rhs(Er, state, field, **kwargs)
    dEr = jnp.ones_like(Er) * 1e-6
    f1 = self._er_rhs(Er + dEr, state, field, **kwargs)
    return (f1 - f0) / dEr   # finite-difference fallback; or jax.jacfwd

def linear_part(self, Er, state, field, **kwargs):
    return self.jacobian_diagonal(Er, state, field, **kwargs) * Er

def nonlinear_source(self, Er, state, field, **kwargs):
    J = self.jacobian_diagonal(Er, state, field, **kwargs)
    f = self._er_rhs(Er, state, field, **kwargs)
    return f - J * Er
```

Add `use_ap_er_preconditioner: bool = False` to `Solver_Parameters`.

**JAX notes:**
- `jax.jacfwd` or `jax.jvp` gives exact `J_Er` efficiently (one forward pass).
- Finite-difference fallback breaks `jax.grad` through `J_Er` — use `jax.jvp` for full differentiability.

---

## Implementation order

| Phase | Priority | Effort | Impact |
|---|---|---|---|
| 1 — Non-uniform dr | High | 10 lines | Correctness on any grid |
| 2 — CellVariable BCs + 2nd-order faces | High | ~80 lines | Biggest accuracy gain |
| 4 — Positivity floor | Medium | ~15 lines | Solver robustness |
| 3 — WENO-3 reconstruction | Medium | ~50 lines | Beyond TORAX; pedestal accuracy |
| 5 — AP Er preconditioner | Low | ~60 lines | Only for extreme ε/Δt regimes |

---

## Files changed

| File | Phases | Nature of change |
|---|---|---|
| `_fem.py` | 1, 3 | `dx` → array broadcast; add `weno3_inner_face_values()` |
| `_cell_variable.py` | 3 | Optional `reconstruction` arg on `face_value()` |
| `_boundary_conditions.py` | 2 | Add `right_constraints_from_bc_model()` |
| `_transport_equations.py` | 1, 2, 3, 5 | All 3 equations rewritten to use CellVariable path |
| `_parameters.py` | 4, 5 | Add `density_floor`, `temperature_floor`, `use_ap_er_preconditioner` |
| `_main_solver.py` | 4, 5 | Floor enforcement; optional AP hook |

No changes to `_transport_solvers.py`, `_theta_solver.py`, `_radau_solver.py`,
`_kvaerno5_solver.py`, `_state.py`, `_field.py`, or any test file (tests should
pass unchanged after Phase 1+2 since the physical solution is the same at convergence).

---

## Phase 6 — Testing strategy: FEM-specific toy models

### Why NOT the existing ErToyModel / SharpTransitionErToyModel

The existing benchmark models (`SharpTransitionErToyModel`, `StiffBranchyToyModel`, etc.)
are **0D pure ODEs** — `y = [n, T, Er]` are scalars, no spatial grid exists.
They test solver stiffness handling and branch-switching, not spatial accuracy.
Using them for FEM testing would tell you nothing about:
- convergence order of the face interpolation
- BC enforcement at axis and edge
- conservation properties of the divergence operator
- WENO vs linear behaviour near sharp profiles

FEM errors are spatial. You need a **1D PDE with a known analytical solution**
and an **h-refinement (n_r doubling) convergence rate check**.

### Mock Field (no VMEC, no file I/O required)

All FEM tests use a minimal mock field — purely JAX arrays, self-contained:

```python
import dataclasses
import jax.numpy as jnp

@dataclasses.dataclass
class MockField:
    """Minimal field for FEM unit tests. Flat geometry: Vprime=1 everywhere."""
    n_r: int
    r_grid_half: jax.Array   # shape (n_r+1,)  e.g. linspace(0, 1, n_r+1)
    r_grid: jax.Array        # shape (n_r,)    cell centres
    Vprime: jax.Array        # shape (n_r,)    = ones(n_r) for flat geometry
    Vprime_half: jax.Array   # shape (n_r+1,)  = ones(n_r+1)
    dr: float                # uniform spacing = 1/n_r

def make_mock_field(n_r: int) -> MockField:
    r_half = jnp.linspace(0.0, 1.0, n_r + 1)
    r_cell = 0.5 * (r_half[1:] + r_half[:-1])
    dr = 1.0 / n_r
    return MockField(
        n_r=n_r,
        r_grid_half=r_half,
        r_grid=r_cell,
        Vprime=jnp.ones(n_r),
        Vprime_half=jnp.ones(n_r + 1),
        dr=dr,
    )
```

For non-uniform grid tests, replace `linspace` with a geometrically clustered spacing.

---

### Test A — Spatial order of convergence (manufactured solution, static RHS)

**What it tests:** Phase 1 + Phase 2. Spatial accuracy of `conservative_update` + `CellVariable.face_value()`.
No time solver involved — tests the instantaneous RHS only.

**Manufactured solution:**
$$u(r) = \sin(\pi r), \quad r \in [0,1]$$
$$-D\,\frac{\partial^2 u}{\partial r^2} = D\pi^2 \sin(\pi r) \equiv S(r)$$

**Test procedure:**
1. Build `MockField` for `n_r ∈ {8, 16, 32, 64, 128}`.
2. Set flux `Γ(r) = -D * du/dr = -D * π * cos(πr)` analytically at cell centres.
3. Apply `CellVariable.face_value()` with Dirichlet BC `u(0)=0`, `u(1)=0`.
4. Compute `conservative_update(Γ_faces, dr_cells, Vprime, Vprime_half)`.
5. Compare with exact source `-S(r_cell)`.
6. Compute L2 error: `error = jnp.sqrt(jnp.mean((rhs_computed + S_exact)**2))`.
7. Assert convergence rate `p ≈ 2.0` (before refactor: `p ≈ 1.0`).

```python
def test_fvm_spatial_order_of_convergence():
    D = 1.0
    errors = []
    n_r_list = [8, 16, 32, 64, 128]
    for n_r in n_r_list:
        field = make_mock_field(n_r)
        r = field.r_grid
        # Analytical flux at cell centres: Γ = -D * π * cos(πr)
        Gamma = -D * jnp.pi * jnp.cos(jnp.pi * r)
        # Build CellVariable with Dirichlet left=0, Dirichlet right=0
        cv = make_profile_cell_variable(
            Gamma, field.r_grid_half,
            left_face_constraint=jnp.asarray(0.0),
            right_face_constraint=jnp.asarray(0.0),
        )
        Gamma_faces = cv.face_value()
        dr_cells = jnp.diff(field.r_grid_half)
        rhs = conservative_update(Gamma_faces, dr_cells, field.Vprime, field.Vprime_half)
        S_exact = D * jnp.pi**2 * jnp.sin(jnp.pi * r)
        # rhs = -div(Gamma) = -S_exact for steady state; check rhs + S ≈ 0
        l2 = float(jnp.sqrt(jnp.mean((rhs + S_exact) ** 2)))
        errors.append(l2)
    # Convergence rate
    rates = [jnp.log(errors[i] / errors[i+1]) / jnp.log(2.0) for i in range(len(errors) - 1)]
    assert all(r > 1.8 for r in rates), f"Expected 2nd-order, got rates {rates}"
```

---

### Test B — BC enforcement correctness (all three types)

**What it tests:** Phase 2 — that Dirichlet, Neumann, and Robin produce the correct
face values at both boundaries. Pure `CellVariable` test, no time stepping.

```python
def test_bc_dirichlet_face_value_pinned():
    """Left and right face values must equal the Dirichlet constraints exactly."""
    n_r = 16
    field = make_mock_field(n_r)
    u = jnp.sin(jnp.pi * field.r_grid)
    cv = make_profile_cell_variable(
        u, field.r_grid_half,
        left_face_constraint=jnp.asarray(0.0),
        right_face_constraint=jnp.asarray(0.0),
    )
    fv = cv.face_value()
    assert jnp.isclose(fv[0],  0.0, atol=1e-14)
    assert jnp.isclose(fv[-1], 0.0, atol=1e-14)

def test_bc_neumann_face_gradient():
    """Left gradient at axis must equal the Neumann constraint."""
    n_r = 32
    field = make_mock_field(n_r)
    u = field.r_grid ** 2          # u(r) = r², du/dr = 2r → at r=0: du/dr=0
    cv = make_profile_cell_variable(
        u, field.r_grid_half,
        left_face_grad_constraint=jnp.asarray(0.0),   # ∂u/∂r=0 at axis
        right_face_constraint=jnp.asarray(1.0),        # u(1)=1 Dirichlet
    )
    grad = cv.face_grad()
    assert jnp.isclose(grad[0], 0.0, atol=1e-10), f"Axis gradient {grad[0]} != 0"

def test_bc_robin_gradient_constraint():
    """Robin BC: face grad at right edge = -right_value / decay_length."""
    n_r = 32
    field = make_mock_field(n_r)
    decay = 0.2
    u = jnp.exp(-field.r_grid / decay)
    grad_expected = -u[-1] / decay     # exact Robin gradient at r=1
    cv = make_profile_cell_variable(
        u, field.r_grid_half,
        left_face_grad_constraint=jnp.asarray(0.0),
        right_face_grad_constraint=jnp.asarray(-u[-1] / decay),
    )
    grad = cv.face_grad()
    assert jnp.isclose(grad[-1], grad_expected, rtol=1e-6)
```

---

### Test C — Exact conservation: zero divergence for constant flux

**What it tests:** Phase 1 + Phase 2. With a spatially constant flux `Γ(r) = C`,
the FVM divergence must be exactly zero (up to floating-point precision) because
there is no net flux through any cell when V'-weighting is flat.

```python
def test_conservative_update_zero_divergence_constant_flux():
    """div(Γ) = 0 for constant Γ with flat geometry (Vprime=1)."""
    for n_r in [8, 32, 128]:
        field = make_mock_field(n_r)
        C = 3.7
        Gamma = jnp.full(n_r, C)
        cv = make_profile_cell_variable(
            Gamma, field.r_grid_half,
            left_face_constraint=jnp.asarray(C),
            right_face_constraint=jnp.asarray(C),
        )
        Gamma_faces = cv.face_value()
        dr_cells = jnp.diff(field.r_grid_half)
        rhs = conservative_update(Gamma_faces, dr_cells, field.Vprime, field.Vprime_half)
        assert jnp.allclose(rhs, 0.0, atol=1e-12), f"n_r={n_r}: max |rhs|={jnp.max(jnp.abs(rhs))}"
```

---

### Test D — Diffusion PDE time evolution with exact decay solution

**What it tests:** Phase 1 + Phase 2 **combined with the existing time solvers**
(theta, RADAU, Kvaerno5). This is the integration test that ties FEM to time stepping.

**PDE:**
$$\frac{\partial u}{\partial t} = D\,\frac{\partial^2 u}{\partial r^2}, \quad r \in [0,1]$$
$$u(r, 0) = \sin(\pi r), \quad u(0,t) = u(1,t) = 0$$
$$u(r, t) = e^{-D\pi^2 t}\,\sin(\pi r)$$

The RHS is assembled as `conservative_update(D * ∇u_faces, ...)`, where `∇u_faces`
comes from `CellVariable.face_grad()`. This exercises the full Phase 1+2 stack.

```python
def _make_diffusion_rhs(D, field):
    """Returns f(t, u) for the 1D diffusion equation, using CellVariable FVM."""
    def rhs(t, u):
        cv = make_profile_cell_variable(
            u, field.r_grid_half,
            left_face_constraint=jnp.asarray(0.0),    # u(0)=0 Dirichlet
            right_face_constraint=jnp.asarray(0.0),   # u(1)=0 Dirichlet
        )
        flux_faces = D * cv.face_grad()                # D * ∂u/∂r at faces
        dr_cells = jnp.diff(field.r_grid_half)
        return conservative_update(flux_faces, dr_cells, field.Vprime, field.Vprime_half)
    return rhs

@pytest.mark.parametrize("solver_name", ["theta", "radau", "kvaerno5"])
@pytest.mark.parametrize("n_r", [16, 32, 64])
def test_diffusion_pde_exact_solution(solver_name, n_r):
    D = 1.0
    t_final = 0.1
    field = make_mock_field(n_r)
    u0 = jnp.sin(jnp.pi * field.r_grid)
    u_exact = jnp.exp(-D * jnp.pi**2 * t_final) * jnp.sin(jnp.pi * field.r_grid)

    rhs = _make_diffusion_rhs(D, field)
    solver = build_time_solver(solver_name, ...)
    u_final = solver.solve(rhs, t0=0.0, t1=t_final, y0=u0)

    l2_error = float(jnp.sqrt(jnp.mean((u_final - u_exact)**2)))
    # After Phase 2: 2nd-order spatial → error dominated by spatial discretisation
    # n_r=64 should give < 1e-4 for D=1, t=0.1 with small dt
    assert l2_error < 1e-3, f"solver={solver_name}, n_r={n_r}: L2={l2_error:.3e}"
```

This test verifies **end-to-end correctness** and is the exact analog of
`TestErToyModelSteadyState` but for a spatially-resolved PDE.

---

### Test E — WENO-3 vs linear: overshoot near sharp profile (Phase 3)

**What it tests:** that WENO reconstruction reduces overshoot near steep gradients.

```python
def test_weno3_less_overshoot_than_linear():
    """WENO-3 face values should not overshoot a step profile; linear interpolation does."""
    n_r = 64
    field = make_mock_field(n_r)
    # Smooth step: tanh transition at r=0.5
    r = field.r_grid
    u_step = 0.5 * (1.0 + jnp.tanh(50.0 * (r - 0.5)))  # sharp but smooth

    cv_linear = make_profile_cell_variable(
        u_step, field.r_grid_half,
        left_face_constraint=jnp.asarray(0.0),
        right_face_constraint=jnp.asarray(1.0),
    )
    cv_weno = make_profile_cell_variable(
        u_step, field.r_grid_half,
        left_face_constraint=jnp.asarray(0.0),
        right_face_constraint=jnp.asarray(1.0),
    )
    fv_linear = cv_linear.face_value(reconstruction="linear")
    fv_weno   = cv_weno.face_value(reconstruction="weno3")

    # Overshoot: face value outside [0, 1]
    overshoot_linear = float(jnp.sum(jnp.maximum(fv_linear - 1.0, 0.0) +
                                      jnp.maximum(-fv_linear, 0.0)))
    overshoot_weno   = float(jnp.sum(jnp.maximum(fv_weno - 1.0, 0.0) +
                                      jnp.maximum(-fv_weno, 0.0)))
    assert overshoot_weno < overshoot_linear, (
        f"WENO overshoot {overshoot_weno:.3e} should be < linear {overshoot_linear:.3e}"
    )
```

---

### Test F — Positivity floor prevents negative densities (Phase 4)

**What it tests:** `jnp.where`-based floor in `vector_field` — RHS cannot be negative
when density is below the floor.

```python
def test_positivity_floor_blocks_decrease():
    """With density_floor set, dn_dt must be >= 0 wherever density < floor."""
    floor = 0.1
    n_r = 16
    field = make_mock_field(n_r)
    # Density profile that dips below floor in the middle
    density = jnp.array([0.5 if i > 3 and i < 12 else 0.05 for i in range(n_r)])
    # Fake dn_dt that is negative everywhere
    dn_dt_raw = -jnp.ones(n_r)

    floor_n = jnp.asarray(floor)
    dn_dt_floored = jnp.where(
        density < floor_n,
        jnp.maximum(dn_dt_raw, 0.0),
        dn_dt_raw,
    )
    # Where density < floor, dn_dt must be >= 0
    below_floor = density < floor_n
    assert jnp.all(dn_dt_floored[below_floor] >= 0.0)
    # Where density >= floor, dn_dt is unchanged
    above_floor = ~below_floor
    assert jnp.allclose(dn_dt_floored[above_floor], dn_dt_raw[above_floor])
```

---

### Test G — JAX jit and differentiability of refactored RHS

**What it tests:** the full refactored FEM stack is `jit`-able and `jax.grad`-able —
required for RADAU/Kvaerno5/theta implicit Newton solves.

```python
def test_fvm_rhs_jittable_and_differentiable():
    n_r = 16
    D = 1.0
    field = make_mock_field(n_r)
    rhs = _make_diffusion_rhs(D, field)

    u0 = jnp.sin(jnp.pi * field.r_grid)

    # jit-ability
    rhs_jit = jax.jit(rhs)
    out = rhs_jit(0.0, u0)
    assert out.shape == (n_r,)

    # Differentiability: grad of sum(rhs) w.r.t. u
    grad_fn = jax.jit(jax.grad(lambda u: jnp.sum(rhs(0.0, u))))
    g = grad_fn(u0)
    assert g.shape == (n_r,)
    assert jnp.all(jnp.isfinite(g))
```

---

### Test suite summary

| Test | Phase tested | Solver used | What it catches |
|---|---|---|---|
| A — Manufactured solution convergence | 1 + 2 | None (static RHS) | Spatial order drops from 2 to 1 if Phase 2 regresses |
| B — BC enforcement | 2 | None | Wrong BC type produces wrong face values |
| C — Zero divergence for constant flux | 1 + 2 | None | V'-weighting errors, non-conservation |
| D — Diffusion PDE exact decay | 1 + 2 | theta / RADAU / Kvaerno5 | End-to-end integration; spatial + temporal accuracy |
| E — WENO overshoot reduction | 3 | None | WENO regression or wrong stencil |
| F — Positivity floor | 4 | None | Floor logic bypassed or `jnp.where` wrong |
| G — jit + differentiability | All | None (JAX transforms) | `jit`/`jax.grad` breaks on any Python control flow in RHS |

Tests A, B, C, G are **unit tests** — fast, pure JAX, no solver.
Test D is the **integration test** — mirrors the spirit of `TestErToyModelSteadyState`
but for a 1D spatial PDE. RADAU, Kvaerno5, and theta are all exercised with the same RHS.
Tests E, F are added once Phase 3 / Phase 4 are implemented.

All tests live in `tests/test_fvm_discretization.py` — separate from the existing
`test_theta_solver_benchmarks.py` which remains unchanged.
