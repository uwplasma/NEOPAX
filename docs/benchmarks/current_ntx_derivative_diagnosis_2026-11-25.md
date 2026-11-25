# Current NTX derivative diagnosis

The original FD capture `outputs/original_fd_ntx_capture_lq73VM` was received
and inspected with NumPy and compiler-text comparisons only. No local JAX
import, equilibrium, coefficient, root or transport solve ran.

## What the original capture establishes

Its 191 saved upstream geometry/channel/native-surface fields match both the
successful and failed archived NTSS1D builds. All 44 geometry/database fields
match the successful archived build. Only the three coefficient tables differ
from the failed archive, with the previously reported gaps. The old failure
did not recur; this is not a correction or a gradient validation.

At surface 4, all 48 prepared leaves, all 880 raw coefficients, and their
types/weakness/device commitments match the standalone replay exactly. Shared
NTX/SOLVAX sources are byte-identical, including `_neopax_ad.py`, `_geometry.py`
and SOLVAX's `direct.py` and `_reproducible.py`.

The actual initialization compiled 70 NTX preparation programs and one scan
program; subsequent surfaces reused them. Of 69 shared StableHLO programs,
68 have identical arithmetic/backend text after removing debug call stacks.
The scan differs only in unroll factors 2 versus 4 for two transpose fusions.
Its values are identical. Those observed compiler differences do not explain
the failed table or the geometry-gradient gap.

## The remaining generic geometry-gradient gap

The current generic AD16/FD16 snapshots allow a finer split using their
actual objective cotangents and geometry/database directions:

| Cost / parameter | Direct geometry gap / abs(full FD) | Database gap / abs(full FD) | Full AD/FD relative difference |
| --- | ---: | ---: | ---: |
| softmax Er / RBC | `2.833763e-09` | `8.376744e-06` | `8.381627e-06` |
| bootstrap current / RBC | `1.636478e-07` | `4.647737e-03` | `4.647559e-03` |
| softmax Er / ZBS | `1.012569e-07` | `1.641751e-05` | `1.650857e-05` |
| bootstrap current / ZBS | `4.539078e-08` | `5.635590e-04` | `5.635140e-04` |

Each component gap is its absolute AD-minus-FD difference divided by the
absolute **full** FD derivative. These are contributions to the full error,
not relative errors using each component as its denominator; cancellation
and the support-chain remainder prevent exact addition. The
[26-row CSV](fixed_projection_generic_table_gradient_split_2026-11-25.csv)
retains signed contributions and both gradients.

For example, the RBC bootstrap direct-geometry gap is `-1.957052e-08`, while
the database gap is `5.558196e-04`. The recorded support adjoint applied to the
saved FD support direction still reproduces the full FD result to
`1.455171e-08` relative. This localizes the dominant discrepancy to the
native-input-to-database derivative or its FD secant. It does not by itself
decide whether the NTX custom derivative or the FD coefficient calculation
is inaccurate.

## Specific source fork to test

NTX's ordinary raw solve calls `_solve_modes_with_tail_residual`, retaining
the lowest three modes after full-tail elimination. Its production custom
pullback calls `_prepared_implicit_vjp_primal`, which recomputes a full
block-Thomas factorization and full fields. Both represent the same discrete
system. Their primal and derivative agreement at the relevant operating
points must be measured before changing either implementation.

The existing selected-node checker now has `--jvp-vjp-only`. It uses all 13
captured transport-objective RHS rows, converting their D11_log/D13/D33 bars
through the actual raw-coefficient normalization. It compares the raw-solve
JVP with the existing production multi-RHS VJP and records both primals.
It runs no FD endpoints or perturbation sweep. Default FD diagnostic scales
remain unchanged. The 10 checker tests and 22 baseline-wrapper tests pass
locally with NumPy/mocks only; GPU validation is pending.

The selected nodes are `(3,4,3)`, a large D11 contribution to softmax Er, and
`(4,4,2)`, a large D13/D33 contribution to bootstrap current in the actual
saved results. They are checked for both RBC and ZBS directions. This tests
local contributions of the actual AD16 objective rows, not a full trajectory
or all database nodes. Passing it cannot establish full AD/FD agreement.

Publish from local WSL:

```bash
(
  cd /mnt/d/Github_Codes/NEOPAX-v0_main || exit 1
  git switch en/v0_main || exit 1
  neopax_files=(
    examples/benchmarks/benchmark_transport_realtime_geometry_forward_fd.py
    examples/benchmarks/check_saved_transport_baseline_repeatability.py
    examples/benchmarks/check_captured_ntx_directional_derivative.py
    tests/test_saved_transport_baseline_repeatability.py
    tests/test_captured_ntx_directional_derivative.py
    docs/benchmarks/ntss1d_transport_ad_fd.md
    docs/benchmarks/current_ntx_derivative_diagnosis_2026-11-25.md
    docs/benchmarks/fixed_projection_generic_table_gradient_split_2026-11-25.csv
  )
  git add -- "${neopax_files[@]}" || exit 1
  git commit --only -m "Check NTX forward and reverse derivatives without FD perturbations" \
    -- "${neopax_files[@]}" || exit 1
  git push origin en/v0_main
) || printf '\nPublish stopped; terminal kept open.\n'
```

Then on the large machine:

```bash
(
  cd "$HOME/NEOPAX" || exit 1
  git switch en/v0_main || exit 1
  git pull --ff-only origin en/v0_main || exit 1
  neopax_case=outputs/fixed_projection_ad_fd16_gu6ZO2/black_box
  neopax_out=$(mktemp -d outputs/ntx_derivative_probe_XXXXXX) || exit 1
  set -o pipefail
  for neopax_parameter in RBC:1:0 ZBS:1:0; do
    neopax_safe=${neopax_parameter//:/_}
    env -u PYTHONPATH -u LD_LIBRARY_PATH -u JAX_PLATFORM_NAME \
      -u XLA_FLAGS -u JAX_COMPILATION_CACHE_DIR -u XLA_PYTHON_CLIENT_ALLOCATOR \
      -u NEOPAX_GRADIENT_AUDIT_DIR -u NEOPAX_ORIGINAL_FORWARD_DIR \
      -u NEOPAX_ORIGINAL_FORWARD_STOP \
      JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=true JAX_ENABLE_COMPILATION_CACHE=0 \
      XLA_PYTHON_CLIENT_PREALLOCATE=false NTX_DISABLE_COMPILATION_CACHE=1 \
      VMEX_COMPILATION_CACHE=disabled \
      "$HOME/jax_env/bin/python" \
      examples/benchmarks/check_captured_ntx_directional_derivative.py \
      --run-dir "$neopax_case" --parameter "$neopax_parameter" \
      --points 3:4:3,4:4:2 --jvp-vjp-only \
      --out-dir "$neopax_out/$neopax_safe" \
      2>&1 | tee "$neopax_out/$neopax_safe.log" || exit 1
  done
  tar -czf outputs/ntx_derivative_probe.tar.gz -C "$neopax_out" . || exit 1
  printf 'Saved: %s\n' "$neopax_out"
) || printf '\nNTX derivative probe stopped; terminal kept open.\n'
```

The NPZs retain all five ordinary/JVP/pullback primal coefficients, raw JVPs,
the 13 coefficient-bar rows and their reverse contractions. CSV `primal`
values in this mode are cotangent-weighted raw coefficients, not transport
objective values. `gradient_fd` and `fd_step` are empty because no perturbation
is evaluated. No numerical method, tolerance or production equality guard is
changed. The existing preflight rejects a reconstructed coefficient primal
that fails its saved-table comparison before comparing derivatives.
