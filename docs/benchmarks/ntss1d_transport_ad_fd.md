# Eight-DOF transport AD/FD comparison with NTSS1D interpolation

The standard `Solve_Transport_equations_wHe_radau.toml` reads a saved database
using `preprocessed_3d_radial_ntss1d`. The recent realtime-geometry AD/FD
benchmarks instead build live NTX tables and use the `generic` interpolation
rule. They therefore do not test the same interpolation formula.

The new input
`examples/benchmarks/Solve_Transport_equations_wHe_radau_ntx_scan_runtime_database_vmec_realtime_geometry_benchmark_ntss1d.toml`
copies the existing black-box AD/FD benchmark and changes only:

```toml
[neoclassical]
interpolation_mode = "preprocessed_3d_radial_ntss1d"
```

It retains the existing live NTX scan, its physical coefficient values and
field nodes, realtime VMEX geometry, initial profiles, sources, 51-point
transport grid and Radau configuration. This compares the interpolation on
the same realtime benchmark; it does not substitute the saved NTSS-comparison
database, equilibrium or other transport settings. The VMEX response remains
`raw_block`, the `v0_new`-style default.

## Forward and derivative contracts

`MonoenergeticRuntimeNTSS1D` retains the same 19 independent database leaves
as the generic live-scan database. Its radial and electric-field coordinate
views supply the existing
`get_Dij_preprocessed_3d_ntss_radius_ntss1d` kernel. No alternate NTSS formula
or duplicate coefficient tables are introduced. The interpolation selection
is propagated through the forward scan conversion and both scalar and
batched recorded scan transposes.

The NTSS1D electric-field interpolation has data-dependent slope limiting.
Its table derivative is obtained by transposing that actual kernel, for
centre fluxes, face fluxes and terminal bootstrap current. A two-knot linear
electric-field stencil would differentiate a different function.

The runner selects `established` for the transport, selected-root and bootstrap
table-transpose boundaries. `legacy_sparse` is specific to the generic
interpolator and is rejected for this new live NTSS1D representation. The
accepted-step map, recorded-forward Newton/Jacobian decisions, segmented
transport reverse and one final recorded NTX scan transpose are retained.

Both AD and FD summaries record `database_interpolation_mode`. FD retains
the frozen-linearized geometry and selected-root lanes and the
`reverse_frozen_metadata` accepted-replay contract.

## Local verification and remote execution

The local tests use NumPy and extracted production code, without JAX imports
or physical solves. They check identical coefficient leaves, derived
coordinates, the sole TOML difference, recorded-scan wiring and forward
agreement with the existing preprocessed NTSS1D representation at 36 queries
spanning radial branches, collisionality clamps and electric-field clamps.

The GPU checks remain to be run on the large machine. The Python runner first
executes three small checks: agreement with the established preprocessed
kernel, coefficient-table pullbacks versus FD, and AD/FD of the complete
scan-conversion/coordinate chain. These checks run no VMEX equilibrium, NTX
coefficient or transport solver. The full benchmark workers start only after
the checks pass.

Then the runner executes one AD worker and eight FD workers sequentially for:

- `n0`, `T0`, `density_shape_power`, `temperature_shape_power`,
  `density_shape_alpha`, `temperature_shape_alpha`;
- `RBC:1:0`, `ZBS:1:0`.

The default is 16 accepted steps, four-step reverse segments, FD relative
increment `3e-7` and absolute increment `1e-10`. Results, logs and snapshots
are copied into separate worker directories. A rerun reuses completed worker
reports when the saved configuration, source digest and increments agree.
Use a new output directory after changing dependencies.

On the large machine, after pushing and pulling the changes:

```bash
(
  cd "$HOME/NEOPAX" || exit 1
  git switch en/v0_main || exit 1
  git pull --ff-only origin en/v0_main || exit 1
  neopax_run=outputs/ntss1d_ad_fd16_20261116
  mkdir -p "$neopax_run" || exit 1
  set -o pipefail

  "$HOME/jax_env/bin/python" \
    examples/benchmarks/benchmark_transport_ntss1d_ad_fd.py \
    --out-dir "$neopax_run" \
    --accepted-step-limit 16 \
    2>&1 | tee "$neopax_run/comparison.log"
) || printf '\nComparison stopped after an error; terminal kept open.\n'
```

`--plan-only` prints all nine benchmark commands without importing JAX,
creating output files or launching workers.

The output `ad_vs_fd.csv` contains 104 transport-gradient pairs. Numerical
columns use scientific notation and relative differences are
`abs(AD-FD)/abs(FD)`. The AD report also retains the geometry-objective rows.
Each FD column is checked against the saved AD initial state, terminal state,
final time, geometry and database arrays. Their maximum absolute differences
and bitwise-match flags are saved. Objective values are checked with a
roundoff allowance of `1e-12*(1+abs(AD_value))`. A
`comparison_baseline_matches=False` row cannot establish an AD derivative
error at a common baseline; inspect the recorded differences first.

No full GPU benchmark or numerical JAX derivative check was run locally.

## GPU preflight failure and interpolation guard fix, 2026-11-16

The remote run at `822607f` stopped in the kernel checks before launching any
transport workers. The forward comparison and scan-conversion/coordinate AD
check passed. The coefficient-table pullback check found a nonfinite result;
the original assertion did not print its coefficient/query subcase.
The [supplied log](ntss1d_preflight_failure_2026-11-16.txt) is retained.

Source-only NumPy execution reproduced unsafe arithmetic for the upper-clamp
query `(radius=1.6, nu/v=0.002, Er/v=0.1)`: `_inpold_fixed` evaluated `0/0`
inside a `where` whose returned value was zero. Evaluating both conditional
branches also exposed nonfinite arithmetic in the inactive interior branch,
where the padded knot was `inf`. These computations can contaminate AD even
when the selected forward value is finite.

The fix guards the denominator before division, replaces only unused padding
with finite increasing coordinates inside the polynomial, and makes inactive
repeated-knot radial weights finite. The existing interpolation formula and
determinant cutoff are retained. Geometry and transport solver code is
unchanged.

The local regression executes both sides of the conditionals with NumPy
floating-point exceptions enabled. Six local interpolation checks pass;
the three GPU checks are skipped locally. Against the pre-fix capture,
[all 108 forward coefficient values](ntss1d_guard_forward_check_2026-11-16.csv)
are bitwise identical. This is forward/arithmetic verification, not a GPU AD
result. The GPU tests now label each coefficient/query subcase and also check
the coordinate chain with batched interior, exact-node and clamped queries.

After pushing/pulling the correction, rerun the Python comparison with a new
directory, for example `outputs/ntss1d_ad_fd16_guard_fix_20261116`. The old
directory records the previous source digest and failed preflight. The runner
again starts transport AD/FD only after its GPU kernel checks pass.
