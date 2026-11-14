# Unresolved Er and bootstrap gradient investigation

The full sixteen-accepted-step discrepancy is **not fixed**. The latest
matched-value RBC bootstrap derivative is `3.8531638809481883e-1` in AD and
`1.1847207619995317e-1` in FD, relative error `2.252381493e0`. The profile
Er differences remain unresolved: the archived profile FD baseline differs
from the latest AD baseline. Agreement of physical replay endpoints does
not certify derivatives, and component tests do not bound error accumulation
through the full calculation.

## Completed isolation checks

Measurements are saved in
[gradient_rule_isolation_2026-11-13.csv](gradient_rule_isolation_2026-11-13.csv).
These are component checks, **not additional full AD16 results**.

* On identical saved NTX prepared inputs, production implicit reverse case
  gradients agree within `3.0e-10` relative under JAX 0.7.2 and 0.10.0
  (they are not bitwise identical), with the same
  current main-based NTX and corrected main-based SOLVAX source. Forward AD
  through the selected-head factorization disagrees with those reverse
  gradients in both versions. For example, the D13 epsilon derivative has
  relative disagreement `1.442349e-3` under 0.10.0 and `5.340295e-4` under
  0.7.2. This inconsistency is not established as the new regression's cause.
* On the saved 51-point plasma state and saved QI database, the terminal
  bootstrap pullback agrees with forward AD: relative differences below
  `6.1e-14` for density, pressure and Er directions, and `2.9e-15` for D13
  and D33 table directions. The D11 direction is zero in both. This fixture
  has a `7x16x12` database, unlike the benchmark's `7x16x11` live database;
  it cannot clear the terminal rule for every point in the remote benchmark.

No equilibrium, initial-root, NTX scan or transport solve was performed for
these checks. The saved NTX single-case solves and terminal evaluations ran
on the local GPU, with bounded executable workspace and no CPU fallback.
No physics, interpolation rule, transport integrator or reverse method was
changed during this investigation.

## Capture the actual full-calculation boundaries

The old logs contain final Jacobian entries but not their intermediate
cotangents. Set this additional variable on the existing AD16 command:

```bash
export NEOPAX_GRADIENT_AUDIT_DIR=outputs/gradient_audit16/ad
```

The optional recording copies existing arrays to unique NPZ files:

* `ad_terminal_*`: actual final packed state, terminal objective cotangents,
  objective labels/values, final time, initial packed state and fixed support.
* `ad_before_scan_fold_*`: profile Jacobian, accumulated database/geometry
  bars and their already-computed components, before the one NTX scan fold.
* `ad_vmec_transpose_input_*` and `ad_vmec_transpose_output_*`: actual VMEC
  state, mask, parameters and full transport state-cotangent rows, followed
  by their parameter cotangents. These use the actual AD16 bars, rather than
  synthetic VMEC test rows.

The flag adds no forward solve, replay, VJP, component scan fold or tangent
solve. In particular, do not enable the wide component-pullback option just
to obtain these files. With the variable unset there is no array traversal
or transfer. Recording does synchronize transfers and uses host/disk space;
the complete remote benchmark with recording has not yet been run.

For FD, use a separate recording directory **for each DOF and run**, while
retaining the original FD step, frozen geometry, selected-root linearization
and accepted-step metadata settings. For example:

```bash
export NEOPAX_GRADIENT_AUDIT_DIR=outputs/gradient_audit16/fd_T0
```

This writes `fd_baseline_*`, `fd_T0_minus_*` and `fd_T0_plus_*`, including
their final packed and physical states. Prioritize T0 for the unresolved
profile root-strength discrepancy and RBC:1:0 for the matched-baseline
bootstrap failure. Existing scalar-only reports cannot supply these arrays
after the fact.

After those runs, this command reads saved files only:

```bash
python examples/benchmarks/compare_transport_gradient_audit.py \
  --ad-dir outputs/gradient_audit16/ad \
  --fd-dir outputs/gradient_audit16/fd_T0 \
  --fd-report outputs/autodiff_transport_lagged_ntx/realtime_geometry_fd/T0_forward_fd_frozen_linearized_summary.json \
  --out outputs/gradient_audit16/T0_terminal_chain.csv
```

It checks the baseline arrays and computes
`terminal_objective_bar @ ((final_y_plus - final_y_minus)/(2*h))`.
For geometry it compares this with the FD report's fixed-geometry final-state
contribution; for profiles it compares with the full objective FD. A profile
disagreement also requires checking any direct dependence of frozen
temperature projections on the profile parameters; it does not automatically
implicate the reverse sweep. The script refuses mismatched baselines instead
of presenting them as current derivative errors. It does not test the entire
NTX/VMEX geometry chain by itself.

Six small array/analytic tests pass for recording, repeated-call preservation,
integer cotangent placeholders, terminal contraction and baseline rejection.
No full local transport validation was run.
