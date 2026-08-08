# wHe NTX Lagged Runtime: Robin Temperature Face Diagnostic

Date: 2026-08-08

## Context

Run under discussion:

```bash
python -m NEOPAX ./examples/benchmarks/Solve_Transport_equations_wHe_radau_ntx_exact_lagged_runtime_vmec_realtime_benchmark.toml
```

This note records the current diagnosis only. It does not imply that solver behavior has been intentionally changed.

## What The Diagnostic Showed

The latest diagnostic output passed beyond the previously suspicious region around `t ~= 0.155`; one later accepted attempt in the attached log was around:

```text
t_start=1.637007e-01 dt_try=1.020099e-03 accepted=True
```

The important temperature distinction was:

```text
state_temperature_center: finite=True min ~= 1.6e-01 keV
face_temperature:         finite=True min = 1.0e-03 keV
```

So the cell-centered temperature state was not zero or NaN. The face temperature was hitting the configured `temperature_floor = 1.0e-3`.

## Confirmed Code Path

Face transport state construction happens in:

```text
NEOPAX/_transport_flux_models.py::build_face_transport_state
```

The relevant sequence is:

```python
temperature_faces = _face_profile(
    state.temperature,
    geometry.r_grid_half,
    bc_model=bc_temperature,
    reconstruction=reconstruction,
)
temperature_faces = safe_temperature(temperature_faces, temperature_floor)
```

Therefore, if the diagnostic reports `face_temperature min=1.0e-03`, that can mean the raw reconstructed face temperature was at or below the floor before clipping.

## Robin Meaning In The Current Code

For the right temperature boundary, the TOML contains:

```toml
[boundary.temperature.right]
type = "robin"

[boundary.temperature.right.value]
default = 1.0

[boundary.temperature.right.decay_length]
default = 0.05
```

The profile-aware right Robin constraint is implemented in:

```text
NEOPAX/_boundary_conditions.py::right_constraints_from_bc_model
```

Current formula:

```python
rv = (4.0 * u_im1 - u_im2) / (3.0 + 2.0 * dx / (decay + 1e-12))
rg = -rv / (decay + 1e-12)
```

This enforces, at the face:

```text
dT/dr = -T_face / L
```

or equivalently:

```text
(dT/dr) / T_face = -1 / L
```

With `L = 0.05`, the implemented log-gradient is `-20`, not `-0.05`.

## Why A Face Can Hit The Floor While Centers Stay Positive

The discrete formula solves the face value from the last two center values:

```text
T_face = (4*T_last - T_prev) / (3 + 2*dx/L)
```

If the edge profile is steep enough that `4*T_last - T_prev` becomes small or negative, the reconstructed face value can become very small or negative even when `T_last` itself is still positive. Then `safe_temperature` clips that face value to `1.0e-3`.

This is the specific contradiction we identified: the continuous Robin/log-gradient condition does not physically require `T=0`, but this discrete reconstruction can permit an invalid positive-state/near-zero-face situation before flooring.

## Comparison Against `en/fix_BCs`

The same profile-aware Robin formula exists in `en/fix_BCs` for:

```text
right_constraints_from_bc_model
apply_cell_centered_boundary_state
```

So this specific Robin formula is not newly introduced by the recent NTX face/center lagged-response refactor. The current diagnostics are exposing that it can become active/problematic in this wHe runtime NTX case.

## Still To Prove

We have not yet printed the raw pre-floor face temperature. The next targeted diagnostic should report:

```text
raw_temperature_faces min/max
first face/species index where raw_temperature_faces <= temperature_floor
whether the index is the right boundary face or an interior face
neighboring center temperatures used in the reconstruction
```

That will distinguish:

- right-boundary Robin reconstruction producing the floored face value
- an interior face reconstruction producing the floored face value
- a species-specific issue

## Possible Fix Direction, Not Yet Applied

If the intended TOML semantics are a Robin decay toward a nonzero boundary/reference value, then the current profile-aware Robin branch likely needs to use `right_value` as the asymptote/reference. A right-face condition like:

```text
dT/dr = -(T_face - T_ref) / L
```

would lead to a different discrete face formula:

```text
T_face = (4*T_last - T_prev + 2*dx*T_ref/L) / (3 + 2*dx/L)
```

But this would change Robin BC semantics globally, so it should only be applied after confirming the intended meaning of `value` and `decay_length` for existing cases.
